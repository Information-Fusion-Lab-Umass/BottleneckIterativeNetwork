# WHAM! Metadata Files

## Types of files
* `scaling npz files` - these are numpy files containing gains used to modify the wsj0 utterances in the original wsj0-2mix dataset. To maintain parity with the original dataset, without porting the voicebox matlab code to python, we include these gains to remove dependency on matlab.
* `mix_params csv files` - contain information on the number of noise samples, before and after the speech signals (for the max version of the dataset), and also the target SNR for the louder speaker.
* `noise_meta csv file` - information on the noise clip for each utterance

## Description of noise_meta fields

* `utterance_id`
* `Noise Band`: Pre-amp gain of the microphones was consistent between all recordings, and files were grouped by their level ranging from 0 (very quiet) to 3 (very loud).  In creating the original WHAM!, care was taken to sample clips uniformly between the four bands, otherwise quieter locations will be over represented.
* `File ID`: All files from the same original recording share this ID
* `L to R Width (cm)`: Horizontal spacing between the two microphones
* `Reverberation Level`: A subjective rating indicating whether the reverberation in the space was high ('h'), medium ('m'), or low ('l')
* `Location ID`: All files recorded in the same location share this ID
* `Location Day ID`: All files recorded at a given location on the same day share this ID