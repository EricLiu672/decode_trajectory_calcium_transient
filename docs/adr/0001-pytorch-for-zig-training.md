# PyTorch for ZIG training

`deconv_calcium_likelihood` now uses PyTorch instead of TensorFlow 1 for fitting the ZIG place-field network. We chose PyTorch because the TF1 session/graph code was no longer installed in the working environment, while PyTorch is the maintained path for the target GPU workflow; the implementation prefers CUDA when a usable torch CUDA backend is available and falls back to CPU when the installed build cannot actually launch kernels on the detected GPU.
