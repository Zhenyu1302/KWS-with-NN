The code is developped based on the speech command tutorial in this link. 
https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html
The tutorial feed raw speech waveform into the convolutional neural network, and thus a 1D convolution is applied. 
However, we feed the fbank feature map into the CNN. Therefore, we modified it into a 2D convolutional layer.

The CNN architecture is built based on the paper listed below.
@article{sainath2015convolutional,
  title={Convolutional neural networks for small-footprint keyword spotting},
  author={Sainath, Tara and Parada, Carolina},
  year={2015}
}
Sainath et al argue that the convolution kernel should cover 2/3 in time, but we keep the orginal setting in the paper even with larger feature map input. 

