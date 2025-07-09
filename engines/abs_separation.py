import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import os
import json
import datetime

import models.afrcnn as afrcnn
import plot.curve as curve
import utils.distributed as distributed_util


class Separation:
    def __init__(self, cfg, result_path):
        self.is_distributed = False
        self.cfg = cfg
        self.result_path = result_path
        self.logger = None
        print("init logger: ")
        self.init_logger()

        self.fusion_model = None
        self.optim = None
        self.loss_func = None
        self.scheduler = None
        self.trained_models = ["fusion_model"]
        self.logger.info("init model: ")
        self.init_model()

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

        self.logger.info("init data: ")
        self.init_data()

    def init_logger(self):
        self.logger = distributed_util.init_logger(is_main=True, is_distributed=False,
                                                   filename=os.path.join(self.result_path, 'run.log')
                                                   )
        self.logger.info(str(self.cfg))

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def init_data(self):
        pass

    def set_train(self):
        for m in self.trained_models:
            getattr(self, m).train()

    def set_eval(self):
        for m in self.trained_models:
            getattr(self, m).eval()

    def is_best_model(self, current, current_best, hist=None):
        return current < current_best

    def save_model(self, criteria, epoch_num):
        state = {
            'epoch': epoch_num,
            'optimizer': self.optim.state_dict(),
            "eval": {}
        }

        for k in criteria:
            state["eval"][k] = criteria[k]

        for k in self.trained_models:
            state[k] = getattr(self, k).state_dict()

        torch.save(state, self.result_path + '/model.pth.tar')

    def load_model(self, checkpoint_path):
        d = torch.load(checkpoint_path)
        self.logger.info("loading model: ")

        for key in d:
            if key not in ["epoch", "optimizer", "eval"]:
                self.logger.info(key)
                getattr(self, key).load_state_dict(d[key])

        return 'best epoch: {}, eval: {}\n'.format(d['epoch'], str(d['eval']))

    @abstractmethod
    def forward_pass(self, input_tuple):
        pass

    def train(self):
        cfg_train = self.cfg["train"]
        batch_size = cfg_train["batch_size"]
        training_progress_filepath = os.path.join(self.result_path, "training_progress.ndjson")
        training_hist_filepath = os.path.join(self.result_path, "training_hist.ndjson")

        best_eval = np.inf
        losses = {"main": []}
        val_losses = {"main": []}
        val_performances = {}

        # train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.train_set, [i for i in range(32)]),
        #                                          batch_size=batch_size)
        # val_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.val_set, [i for i in range(32)]),
        #                                          batch_size=batch_size)

        for i in tqdm(range(int(cfg_train['max_epoch']))):
            self.set_train()
            epoch_begin_time = datetime.datetime.now()

            loss_epoch = {"main": 0}

            self.logger.info("#" * 10 + "epoch {} training".format(i))

            # train the model
            if "need_train" not in cfg_train or cfg_train["need_train"]:
                for batch_idx, input_tuple in enumerate(self.train_loader):
                    start_time = datetime.datetime.now()

                    if self.is_distributed:
                        self.logger.info(
                            f"Local Rank {self.args.local_rank}: Batch {batch_idx} / {len(self.train_loader)}")
                    else:
                        self.logger.info(f"Batch {batch_idx} / {len(self.train_loader)}")

                    self.optim.zero_grad()

                    output, ground_truth, loss = self.forward_pass(input_tuple)
                    # loss is a dict, loss["main"] is the overall loss

                    main_loss = loss["main"]

                    main_loss.backward()
                    self.optim.step()

                    if self.is_distributed:
                        self.logger.info(
                            f"Local Rank {self.args.local_rank}: Model batch time duration: {datetime.datetime.now() - start_time}")
                    else:
                        self.logger.info(f"Model batch time duration: {datetime.datetime.now() - start_time}")
                    for key in loss:
                        if key in loss_epoch:
                            loss_epoch[key] += loss[key].item()
                        else:
                            loss_epoch[key] = loss[key].item()
                if self.is_distributed:
                    self.logger.info(
                        f"Local Rank {self.args.local_rank}: Model epoch {i} training time duration: {datetime.datetime.now() - epoch_begin_time}")
                else:
                    self.logger.info(
                        f"Model epoch {i} training time duration: {datetime.datetime.now() - epoch_begin_time}")

                for key in loss_epoch:
                    loss_epoch[key] /= len(self.train_loader)

                for key in loss_epoch:
                    if key in losses:
                        losses[key].append(float(loss_epoch[key]))
                    else:
                        losses[key] = [float(loss_epoch[key])]

            """
            All the evaluation, model saving and plotting should only happen on local_rank 0 if is_distributed is True
            """
            if not self.is_distributed or (self.is_distributed and self.args.local_rank == 0):
                # validate the model
                val_begin_time = datetime.datetime.now()
                self.set_eval()

                val_loss, val_performance = self.evaluate(self.val_loader)
                # val_loss shares the same format as loss in training process
                # val_performances is dict of multiple additional evaluation metrics

                for key in val_loss:
                    if key in val_losses:
                        val_losses[key].append(float(val_loss[key]))
                    else:
                        val_losses[key] = [float(val_loss[key])]

                for key in val_performance:
                    if key in val_performances:
                        val_performances[key].append(float(val_performance[key]))
                    else:
                        val_performances[key] = [float(val_performance[key])]

                # save the best model
                if self.is_best_model(val_loss["main"], best_eval):
                    self.save_model({"val_loss": val_loss, "val_performance": val_performance}, i)
                    best_eval = val_loss["main"]

                self.logger.info(
                    f"Model epoch {i} validation time duration: {datetime.datetime.now() - val_begin_time}")

                self.logger.info("train loss: {}, val loss: {}, val performance: {}".format(
                    str(loss_epoch),
                    str(val_loss),
                    str(val_performance)))

                # plot training losses and validation loss up to the current epoch
                for key in val_performances:
                    curve.single_plot_one_curve(np.arange(i + 1), val_performances[key], "epoch", key,
                                                self.result_path + "/val_{}.png".format(key))

                train_loss_xs = [range(i + 1) for _ in losses]
                train_legends = list(losses.keys())
                train_loss_ys = [losses[key] for key in train_legends]
                curve.single_plot_multi_curves(train_loss_xs, train_loss_ys, "epoch", "loss", train_legends,
                                               self.result_path + "/train_loss.png")

                val_loss_xs = [range(i + 1) for _ in val_losses]
                val_legends = list(val_losses.keys())
                val_loss_ys = [val_losses[key] for key in val_legends]
                curve.single_plot_multi_curves(val_loss_xs, val_loss_ys, "epoch", "loss", val_legends,
                                               self.result_path + "/val_loss.png")

                # record train & val performance
                with open(training_progress_filepath, "a") as f:
                    cur_report = {"epoch": i}
                    for key in loss_epoch:
                        cur_report["train_{}_loss".format(key)] = float(loss_epoch[key])
                    for key in val_loss:
                        cur_report["val_{}_loss".format(key)] = float(val_loss[key])
                    json.dump(cur_report, f)
                    f.write("\n")

                with open(training_hist_filepath, "w") as f:
                    json.dump({"train_losses": losses, "val_losses": val_losses, "val_performances": val_performances},
                              f, indent=4)

                # update learning rate
                if self.scheduler is not None:
                    before_lr = self.optim.param_groups[0]["lr"]
                    # self.scheduler.step(loss_epoch["main"])
                    self.scheduler_step()
                    after_lr = self.optim.param_groups[0]["lr"]
                    self.logger.info("Epoch %d: Adam lr %.4f -> %.4f" % (i, before_lr, after_lr))

                # test per 10 epoch
                if (i + 1) % 10 == 0:
                    test_begin_time = datetime.datetime.now()
                    self.logger.info("#" * 5 + "Intermediate test")
                    test_loss, test_performance = self.test()
                    self.logger.info("test loss: ", test_loss)
                    self.logger.info("test performance:", test_performance)

                    with open(os.path.join(self.result_path, "test_result_epoch{}.json".format(i)), "w") as f:
                        json.dump({"loss": {key: float(test_loss[key]) for key in test_loss},
                                   "performance": {key: float(test_performance[key]) for key in test_performance}}, f)

                    self.logger.info(
                        f"Model epoch {i} test time duration: {datetime.datetime.now() - test_begin_time}")

    def scheduler_step(self):
        self.scheduler.step()

    def test(self):
        cp_state = self.load_model(
            self.result_path + '/model.pth.tar'
        )

        self.set_eval()

        self.logger.info("#" * 10 + "trained model loaded")
        self.logger.info(cp_state)

        # test_loader = torch.utils.data.DataLoader(self.test_set,
        #                                           batch_size=batch_size,
        #                                           pin_memory=True)
        # test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.test_set, [i for i in range(32)]),
        #                                          batch_size=batch_size)
        return self.evaluate(self.test_loader)

    @abstractmethod
    def evaluate(self, data_loader):
        pass


class DistributedSeparation:
    def __init__(self, cfg, result_path, console_args):
        self.cfg = cfg
        self.args = console_args
        self.result_path = result_path
        self.device = torch.device(f'cuda:{self.args.local_rank}')
        self.logger = None
        print("init logger: ")
        self.init_logger()
        self.logger.debug(f"current local rank: {self.args.local_rank}")

        self.fusion_model = None
        self.optim = None
        self.loss_func = None
        self.scheduler = None

        self.aux_loss_name = "pesq"

        self.trained_models = ["fusion_model"]
        self.logger.info("init model: ")
        self.init_model()

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

        self.train_sampler = None
        self.test_sampler = None
        self.val_sampler = None

        self.logger.info("init data: ")
        self.init_data()

    def init_logger(self):
        self.logger = distributed_util.init_logger(is_main=True, is_distributed=True,
                                                   filename=os.path.join(self.result_path, 'run.log')
                                                   )
        self.logger.info(str(self.cfg))

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def init_data(self):
        pass

    def set_train(self):
        for m in self.trained_models:
            getattr(self, m).train()

    def set_eval(self):
        for m in self.trained_models:
            getattr(self, m).eval()

    def is_best_model(self, current, current_best, hist=None):
        return current < current_best

    def save_model(self, criteria, epoch_num):
        state = {
            'epoch': epoch_num,
            'optimizer': self.optim.state_dict(),
            "eval": {}
        }

        for k in criteria:
            state["eval"][k] = criteria[k]

        for k in self.trained_models:
            state[k] = getattr(self, k).module.state_dict()

        torch.save(state, self.result_path + '/model.pth.tar')

    def save_current_model(self, best_val, epoch_num):
        state = {
            'epoch': epoch_num,
            'optimizer': self.optim.state_dict(),
            "eval": best_val
        }

        for k in self.trained_models:
            state[k] = getattr(self, k).module.state_dict()

        torch.save(state, self.result_path + '/temp_model.pth.tar')

    def load_model(self, checkpoint_path):
        d = torch.load(checkpoint_path)
        self.logger.info("loading model: ")

        if "best_val" in d:
            best_val = d["best_val"]
            d["eval"] = best_val
            d.pop('best_val', None)

        for key in d:
            if key not in ["epoch", "optimizer", "eval"]:
                self.logger.info(key)
                getattr(self, key).module.load_state_dict(d[key])
        
        self.optim.load_state_dict(d["optimizer"])
        return d["eval"], d["epoch"]

    @abstractmethod
    def forward_pass(self, input_tuple):
        pass

    def train(self):
        num_gpus = self.args.n_gpu_per_node

        # tools used to merge distributed gpu outcomes
        def merge_hist(gpu_result, merged_result):
            for key in gpu_result:
                if key not in merged_result:
                    merged_result[key] = gpu_result[key]
                else:
                    for idx, v in enumerate(gpu_result[key]):
                        merged_result[key][idx] += v

        def average_merge(merged_result):
            for key in merged_result:
                for idx, v in enumerate(merged_result[key]):
                    merged_result[key][idx] = v / num_gpus

        cfg_train = self.cfg["train"]
        batch_size = cfg_train["batch_size"]

        best_eval = np.inf

        # train & validation historical performance computed from each gpu
        # need to save them to json files and synchronize them
        losses = {"main": []}
        val_losses = {"main": []}
        val_performances = {}

        training_progress_filepath = os.path.join(self.result_path, "training_progress.ndjson")

        # check if there exists a pretraind model
        if os.path.isfile(self.result_path + '/temp_model.pth.tar'):
            start_eval, start_epoch = self.load_model(self.result_path + '/temp_model.pth.tar')
            best_eval = start_eval
        elif os.path.isfile(self.result_path + '/model.pth.tar'):
            start_eval, start_epoch = self.load_model(self.result_path + '/model.pth.tar')
            best_eval = start_eval["val_loss"]
        else:
            start_epoch = -1

        self.logger.info(f"Local Rank {self.args.local_rank}: start from epoch {start_epoch+1}")

        for i in tqdm(range(start_epoch+1, int(cfg_train['max_epoch']))):
        # for i in tqdm(range(int(cfg_train['max_epoch']))):
            self.set_train()
            epoch_begin_time = datetime.datetime.now()

            loss_epoch = {"main": 0}

            self.logger.debug("#" * 10 + "epoch {} training".format(i))

            # train the model
            if "need_train" not in cfg_train or cfg_train["need_train"]:
                for batch_idx, input_tuple in enumerate(self.train_loader):
                    start_time = datetime.datetime.now()
                    self.logger.info(f"Local Rank {self.args.local_rank}: Batch {batch_idx} / {len(self.train_loader)}")

                    self.optim.zero_grad()

                    output, ground_truth, loss = self.forward_pass(input_tuple)
                    # loss is a dict, loss["main"] is the overall loss

                    main_loss = loss["main"]

                    main_loss.backward()
                    self.optim.step()

                    self.logger.info(
                        f"Local Rank {self.args.local_rank}: Model batch time duration: {datetime.datetime.now() - start_time}")

                    for key in loss:
                        if key in loss_epoch:
                            loss_epoch[key] += loss[key].item()
                        else:
                            loss_epoch[key] = loss[key].item()

                self.logger.info(
                    f"Local Rank {self.args.local_rank}: Model epoch {i} training time duration: {datetime.datetime.now() - epoch_begin_time}")

                for key in loss_epoch:
                    loss_epoch[key] /= len(self.train_loader)

                # record the current epoch training performance
                for key in loss_epoch:
                    if key in losses:
                        losses[key].append(float(loss_epoch[key]))
                    else:
                        losses[key] = [float(loss_epoch[key])]

            # save current train losses in files
            training_hist_filepath = os.path.join(self.result_path, f"training_hist_{self.args.local_rank}.ndjson")
            with open(training_hist_filepath, "w") as f:
                json.dump(losses, f, indent=4)  # {"main": [loss_value_epoch0_device0, loss_value_epoch1_device0, ...]}

            # validate and save current val losses in files
            val_begin_time = datetime.datetime.now()
            self.set_eval()

            # record the current epoch val performance
            val_loss, val_performance = self.evaluate(self.val_loader)

            for key in val_loss:
                if key in val_losses:
                    val_losses[key].append(float(val_loss[key]))
                else:
                    val_losses[key] = [float(val_loss[key])]

            for key in val_performance:
                if key in val_performances:
                    val_performances[key].append(float(val_performance[key]))
                else:
                    val_performances[key] = [float(val_performance[key])]

            val_hist_loss_filepath = os.path.join(self.result_path,
                                                  "val_hist_loss_{}.ndjson".format(self.args.local_rank))
            val_hist_performance_filepath = os.path.join(self.result_path,
                                                         "val_hist_performance_{}.ndjson".format(self.args.local_rank))

            train_hist_merge_path = os.path.join(self.result_path, f"training_hist.ndjson")
            val_hist_loss_merge_filepath = os.path.join(self.result_path,
                                                        "val_hist_loss.ndjson")
            val_hist_performance_merge_filepath = os.path.join(self.result_path,
                                                               "val_hist_performance.ndjson")

            # save current val losses & performances in files
            with open(val_hist_loss_filepath, "w") as f:
                json.dump(val_losses, f,
                          indent=4)  # {"main": [loss_value_epoch0_device0, loss_value_epoch1_device0, ...]}

            with open(val_hist_performance_filepath, "w") as f:
                json.dump(val_performances, f, indent=4)  # {"PESQ": [pesq_epoch0_device0, pesq_epoch1_device0, ...]}

            # wait all process to finish the distributed validation
            self.logger.debug(f"Local Rank {self.args.local_rank}: wait for synchronization. Task: validation")
            torch.distributed.barrier()

            # evaluate using the saved files, val_hist_filepath and train_hist_filepath
            # only happens on local_rank 0
            if self.args.local_rank == 0:
                # validate the model
                # val_loss shares the same format as loss in training process
                # val_performances is dict of multiple additional evaluation metrics

                # merged results used for model saving and plot
                train_hist_merge = {}  # {"main": [loss_value_epoch0_device0, loss_value_epoch1_device0, ...]}
                val_hist_loss_merge = {}  # {"main": [loss_value_epoch0_device0, loss_value_epoch1_device0, ...]}
                val_hist_performance_merge = {}  # {"PESQ": [pesq_epoch0_device0, pesq_epoch1_device0, ...]}

                # merge all gpu results
                for r in range(num_gpus):
                    train_hist = json.load(open(os.path.join(self.result_path, f"training_hist_{r}.ndjson"), "r"))
                    val_hist_loss = json.load(open(os.path.join(self.result_path, f"val_hist_loss_{r}.ndjson"), "r"))
                    val_hist_performance = json.load(open(os.path.join(self.result_path,
                                                                       f"val_hist_performance_{r}.ndjson"), "r"))

                    merge_hist(train_hist, train_hist_merge)
                    merge_hist(val_hist_loss, val_hist_loss_merge)
                    merge_hist(val_hist_performance, val_hist_performance_merge)

                average_merge(train_hist_merge)
                average_merge(val_hist_loss_merge)
                average_merge(val_hist_performance_merge)

                with open(train_hist_merge_path, "w") as f:
                    json.dump(train_hist_merge, f, indent=4)

                with open(val_hist_loss_merge_filepath, "w") as f:
                    json.dump(val_hist_loss_merge, f, indent=4)

                with open(val_hist_performance_merge_filepath, "w") as f:
                    json.dump(val_hist_performance_merge, f, indent=4)

                current_val_loss = val_hist_loss_merge["main"][-1]

                # save the best model
                if self.is_best_model(current_val_loss, best_eval):
                    self.save_model({"val_loss": current_val_loss,
                                     "val_performance": {key: val_hist_performance_merge[key][-1]
                                                         for key in val_hist_performance_merge}}, i)
                    best_eval = current_val_loss

                # save the current model
                self.save_current_model(best_eval, i)

                self.logger.info(
                    f"Model epoch {i} validation time duration: {datetime.datetime.now() - val_begin_time}")

                self.logger.info("train loss: {}, val loss: {}, val performance: {}".format(
                    str({key: train_hist_merge[key][-1] for key in train_hist_merge}),
                    str({key: val_hist_loss_merge[key][-1] for key in val_hist_loss_merge}),
                    str({key: val_hist_performance_merge[key][-1] for key in val_hist_performance_merge})))

                # plot training losses and validation loss up to the current epoch
                for key in val_hist_performance_merge:
                    curve.single_plot_one_curve(np.arange(start_epoch+1, i + 1), val_hist_performance_merge[key], "epoch", key,
                                                self.result_path + "/val_{}.png".format(key))

                train_loss_xs = [range(start_epoch+1, i + 1) for _ in train_hist_merge]
                train_legends = list(train_hist_merge.keys())
                train_loss_ys = [train_hist_merge[key] for key in train_legends]
                curve.single_plot_multi_curves(train_loss_xs, train_loss_ys, "epoch", "loss", train_legends,
                                               self.result_path + "/train_loss.png")

                val_loss_xs = [range(start_epoch+1, i + 1) for _ in val_hist_loss_merge]
                val_legends = list(val_hist_loss_merge.keys())
                val_loss_ys = [val_hist_loss_merge[key] for key in val_legends]
                curve.single_plot_multi_curves(val_loss_xs, val_loss_ys, "epoch", "loss", val_legends,
                                               self.result_path + "/val_loss.png")

                # record train & val performance
                with open(training_progress_filepath, "a") as f:
                    cur_report = {"epoch": i}
                    for key in loss_epoch:
                        cur_report["train_{}_loss".format(key)] = float(train_hist_merge[key][-1])
                    for key in val_loss:
                        cur_report["val_{}_loss".format(key)] = float(val_hist_loss_merge[key][-1])
                    json.dump(cur_report, f)
                    f.write("\n")

            # let other gpus wait for local_rank 0
            self.logger.debug(f"Local Rank {self.args.local_rank}: wait for synchronization. "
                             f"Task: cuda0 summarize validation performance")
            torch.distributed.barrier()

            # update learning rate
            if self.scheduler is not None:
                before_lr = self.optim.param_groups[0]["lr"]
                self.scheduler_step(loss_epoch["main"])
                # self.scheduler.step()
                after_lr = self.optim.param_groups[0]["lr"]
                self.logger.info("Epoch %d: Adam lr %.4f -> %.4f" % (i, before_lr, after_lr))

            self.logger.debug(f"Local Rank {self.args.local_rank}: wait for synchronization. "
                              f"Task: update learning rate")
            torch.distributed.barrier()

            # test per 10 epoch
            if (i + 1) % 10 == 0:
                self.logger.debug("#" * 5 + f"{self.args.local_rank}: Intermediate test")
                test_result_path = self.test(train_epoch=i)
                if self.args.local_rank == 0:
                    test_result = json.load(open(test_result_path, "r"))
                    self.logger.info("test loss: " + str(test_result["loss"]))
                    self.logger.info("test performance:" + str(test_result["performance"]))

            # let other gpus wait for local_rank 0
            self.logger.debug(f"Local Rank {self.args.local_rank}: wait for synchronization. "
                             f"Task: wrap up the whole epoch")
            torch.distributed.barrier()

    def scheduler_step(self, cur_loss):
        self.scheduler.step(cur_loss)

    def test(self, train_epoch=None):
        test_begin_time = datetime.datetime.now()

        cp_state, epoch = self.load_model(
            self.result_path + '/model.pth.tar'
        )

        self.set_eval()

        self.logger.debug("#" * 10 + f"{self.args.local_rank} Test mode: trained model loaded")
        self.logger.info(cp_state)

        test_loss, test_performance = self.evaluate(self.test_loader)

        if train_epoch is not None:
            test_filepath = os.path.join(self.result_path,
                                         "test_result_Tepoch{}_Bepoch{}_gpu{}.json".format(
                                             train_epoch, epoch, self.args.local_rank))
        else:
            test_filepath = os.path.join(self.result_path,
                                         "test_result_Bepoch{}_gpu{}.json".format(epoch, self.args.local_rank))

        with open(test_filepath, "w") as f:
            json.dump({"loss": {key: float(test_loss[key]) for key in test_loss},
                       "performance": {key: float(test_performance[key]) for key in test_performance}},
                      f)

        # synchronize
        self.logger.debug(f"Local Rank {self.args.local_rank}: wait for synchronization. Task: test")
        torch.distributed.barrier()

        # merge
        test_merge = {"loss": {}, "performance": {}}

        if train_epoch is not None:
            test_merge_filepath = os.path.join(self.result_path,
                                               "test_result_Tepoch{}_Bepoch{}.json".format(
                                                   train_epoch, epoch))
        else:
            test_merge_filepath = os.path.join(self.result_path,
                                               "test_result_Bepoch{}.json".format(epoch))

        if self.args.local_rank == 0:
            for r in range(self.args.n_gpu_per_node):
                if train_epoch is not None:
                    test_gpu_filepath = os.path.join(self.result_path,
                                                     "test_result_Tepoch{}_Bepoch{}_gpu{}.json".format(
                                                         train_epoch, epoch, r))
                else:
                    test_gpu_filepath = os.path.join(self.result_path,
                                                     "test_result_Bepoch{}_gpu{}.json".format(epoch, r))
                test_gpu = json.load(open(test_gpu_filepath, "r"))

                for key in test_gpu["loss"]:
                    if key not in test_merge["loss"]:
                        test_merge["loss"][key] = test_gpu["loss"][key]
                    else:
                        test_merge["loss"][key] += test_gpu["loss"][key]

                for key in test_gpu["performance"]:
                    if key not in test_merge["performance"]:
                        test_merge["performance"][key] = test_gpu["performance"][key]
                    else:
                        test_merge["performance"][key] += test_gpu["performance"][key]

            for key in test_merge["loss"]:
                test_merge["loss"][key] /= self.args.n_gpu_per_node
            for key in test_merge["performance"]:
                test_merge["performance"][key] /= self.args.n_gpu_per_node

            with open(test_merge_filepath, "w") as f:
                json.dump(test_merge, f)

            self.logger.info(
                f"test time duration: {datetime.datetime.now() - test_begin_time}")

        # synchronize
        self.logger.debug(f"Local Rank {self.args.local_rank}: wait for synchronization. "
                         f"Task: cuda0 summarize test performance")
        torch.distributed.barrier()

        return test_merge_filepath

    def eval_test(self):
        cp_state, epoch = self.load_model(
            self.result_path + '/model.pth.tar'
        )

        self.set_eval()

        self.logger.debug("#" * 10 + f"{self.args.local_rank} Test mode: trained model loaded")
        self.logger.info(cp_state)

        self.evaluate(self.test_loader)


    @abstractmethod
    def evaluate(self, data_loader):
        pass
