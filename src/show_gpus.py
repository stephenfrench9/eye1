from tensorflow.python.client import device_lib
from keras import backend

if __name__ == '__main__':

    print("LOCAL DEVICES: ")
    print(device_lib.list_local_devices())
    print()
    print()
    print("GPUS: ")
    print(backend.tensorflow_backend._get_available_gpus())
    print()
    print()
    # confirm TensorFlow sees the GPU
    assert 'GPU' in str(device_lib.list_local_devices())
    # confirm Keras sees the GPU
    assert len(backend.tensorflow_backend._get_available_gpus()) > 0