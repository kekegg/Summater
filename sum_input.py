"""Functions for downloading and reading MNIST data."""
from __future__ import print_function
import gzip
import os
import urllib
import numpy
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
def maybe_download(filename, work_directory):
  """Download the data from Yann's website, unless it's already here."""
  if not os.path.exists(work_directory):
    os.mkdir(work_directory)
  filepath = os.path.join(work_directory, filename)
  if not os.path.exists(filepath):
    filepath, _ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  return filepath
def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]
def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data
def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot
def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels)
    return labels
class data(object):
  def __init__(self, num):
    self.num = num
    self.pos = 0



class DataSet(object):
  def __init__(self, images, labels, dim, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
      images = images.astype(numpy.float32)
      images = numpy.multiply(images, 1.0 / 255.0)
      my_nums = []
      for i in range(10):
        dt = images[labels==i]
        my_nums.append(data(dt))
        print(dt.shape)

      images1 = numpy.zeros((1,images.shape[1]))
#      print images.shape
#      print my_nums[0].num[range(dim)].shape
      images2 = numpy.zeros((1,images.shape[1]))
      sum_image = numpy.zeros((1,images.shape[1]))
      for i in range(10): #0..9
        for j in range(0,9-i+1): #0..9-i;  i=8, j=0,1,
#          tmp = numpy.concatenate((my_nums[i].num[range(dim)],my_nums[j].num[range(dim)]),axis=1)
#          print(i)
          images1 = numpy.append(images1, my_nums[i].num[range(dim)],axis=0)
          images2 = numpy.append(images2, my_nums[j].num[range(dim)],axis=0)
          sum_image = numpy.append(sum_image, my_nums[i+j].num[range(dim)],axis=0)

      images1 = numpy.delete(images1,0,0)
      images2 = numpy.delete(images2,0,0)
      sum_image = numpy.delete(sum_image,0,0)

      assert images1.shape[0] == sum_image.shape[0], (
          "images.shape: %s labels.shape: %s" % (images1.shape,
                                                 sum_image.shape))
      self._num_examples = images1.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      # Convert from [0, 255] -> [0.0, 1.0].

      assert images2.shape[0] == sum_image.shape[0], (
          "images.shape: %s labels.shape: %s" % (images2.shape,
                                                 sum_image.shape))
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
#      images2 = images2.reshape(images2.shape[0],
#                              images2.shape[1] * images2.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
    
#      sum_image = sum_image.reshape(sum_image.shape[0],
#                              sum_image.shape[1] * sum_image.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].

    self._images1 = images1
    self._images2 = images2
    self._sum_image= sum_image
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images1(self):
    return self._images1
  @property
  def images2(self):
    return self._images2
  @property
  def sum_image(self):
    return self._sum_image
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(784)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images1 = self._images1[perm]
      self._images2 = self._images2[perm]
      self._sum_image = self._sum_image[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images1[start:end],  self._images2[start:end], self._sum_image[start:end]

def read_data_sets(train_dir, fake_data=False, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()
  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True)
    data_sets.validation = DataSet([], [], fake_data=True)
    data_sets.test = DataSet([], [], fake_data=True)
    return data_sets
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000
  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)
  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)
  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)
  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)

#  validation_images = train_images[:VALIDATION_SIZE]
#  validation_labels = train_labels[:VALIDATION_SIZE]

  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]
  data_sets.train = DataSet(train_images, train_labels, 2000)
# data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels, 400)
  return data_sets
