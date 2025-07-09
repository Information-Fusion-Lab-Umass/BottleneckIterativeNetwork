avlit_default = dict(num_sources=2,

                     # Audio branch
                     kernel_size=40,
                     audio_num_blocks=8,
                     audio_hidden_channels=512,
                     audio_bottleneck_channels=128,
                     audio_states=5,

                     # Video branch
                     video_num_blocks=4,
                     video_hidden_channels=128,
                     video_bottleneck_channels=128,
                     video_states=5,
                     video_embedding_dim=1024,

                     # AV fusion
                     fusion_operation="sum",
                     fusion_positions=[0])


avlit_late = dict(num_sources=2,
                     # Audio branch
                     kernel_size=40,
                     audio_num_blocks=8,
                     audio_hidden_channels=512,
                     audio_bottleneck_channels=128,
                     audio_states=5,

                     # Video branch
                     video_num_blocks=4,
                     video_hidden_channels=128,
                     video_bottleneck_channels=128,
                     video_states=5,
                     video_embedding_dim=1024,

                     # AV fusion
                     fusion_operation="sum",
                     fusion_positions=[7])


avlit_reduce = dict(num_sources=2,
                    # Audio branch
                    kernel_size=40,
                    audio_hidden_channels=512,
                    audio_bottleneck_channels=128,
                    audio_num_blocks=8,
                    audio_states=5,
                    # Video branch
                    video_hidden_channels=128,
                    video_bottleneck_channels=128,
                    video_num_blocks=4,
                    video_states=5,
                    video_encoder_checkpoint="",
                    video_encoder_trainable=False,
                    video_embedding_dim=256,
                    # AV fusion
                    fusion_operation="sum",
                    fusion_positions=None)

PermInvariantSISDR_Train = dict(n_sources=2,
                                zero_mean=True,
                                backward_loss=True,
                                return_individual_results=False)

PermInvariantSISDR_Eval = dict(n_sources=2,
                               zero_mean=True,
                               backward_loss=False,
                               return_individual_results=True)
