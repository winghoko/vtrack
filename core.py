'''
Core utilities for simple video motion analysis

## Functions defined in this module

Broadly speaking, functions defined in this module (excluding those intended
for "internal" use) can be classified into four categories as follows:

    1. Image processing related
    'build_grayscaled', 'build_postprocessed', 'build_subtracted', 
    'build_thresholded', 'iter_grayscaled', 'iter_postprocess',
    'iter_postprocessed', 'iter_subtracted', 'iter_thresholded', 
    'calc_background', 'calc_threshold', 'count_intensity'

    2. Image region/path identification related
    'calc_pixel_to_phys', 'convert_to_physical', 'convert_to_pixel',
    'cropbox_to_slices', 'cropbox_to_vertices', 'mark_image', 
    'vertices_to_cropbox', 'vertices_to_mask', 'vertices_to_path'

    3. Analysis related
    'assemble_motion_data', 'compute_centroids', 'compute_n_centroids', 
    'calculate_motion', 'detect_slowness', 'estimate_motion'

    4. I/O related
    'build_from_iter',
    'export_image', 'export_video', 'export_overlaid_video', 
    'export_overlaid_n_video', 'imshow', 'overlay_coords', 'overlay_path', 
    'plot_path', 'plot_n_paths', 'read_csv', 'write_csv'

## Initial Image processing pipeline

The initial image processing pipeline starts with an imageio reader object 
(representing a video or a collation of images) and goes through the steps
initial => grayscale => subtract => threshold [=> postprocess], where the
last step is optional. Functions in the pipeline have both an iterator ("lazy")
version with names "iter..." and normal function ("eager") version with names 
"build...". The iterator versions are (with 1 exception, see below) always 
cumulative and start with the reader object, while the normal function 
versions are always incremental and start with the previous step. Incidentally,
the normal function versions accept either numpy arrays or iterators as input.

A third set of function in this pipeline has names "calc...", which calculate
important parameters needed for the pipeline. These are incremental.

The only exception to the above is iter_postprocess, which is lazy but 
incremental, and takes an iterator to yield frames one by one (however, there 
is also the cumulative iter_postprocessed).

## Computing centroids of foreground object(s) from binary frames

For video with a single foreground object, the location of the object (a.k.a.
its centroid) is calculated by taking the intensity moment. This computation 
is implemented in compute_centroids().

For video with multiple (but fixed number of) foreground objects, the location
of the objects are calculated by feeding the collection of foreground pixels
into a k-means clustering algorithm (more precisely the "mini-batch" variation
of the algorithm), treating each foreground pixel as a data point with equal 
weight. Then, the centroids from the current frame is matched to the centroids
from the previous frame by minimizing the sum of distances from the previous
frame to the current frame. This computation is implemented in 
compute_n_centroids().

Currently this module contains no algorithms to compute the centroids of 
multiple foreground objects when the number of objects vary across frames.

## Analysis of motion from centroids

The main tool for analyzing the centroids obtained from the previous step is 
packed into the function assemble_motion_data(). By default, this function
computes a smoothened version of centroid(s) coordinates and instantaneous 
object velocities using a rolling least-square fit to linear functions. The 
assemble_motion_data() function can work with outputs from both 
compute_centroids() and compute_n_centroids().

## Visualizing and storing results

For single foreground object, the trajectory of the object can be plotted with
plot_path(), and the computed centroid/trajectory can be visualized alongside 
the original video using export_overlaid_video(). The multiple-foreground-
object counterparts to these two functions are plot_n_paths() and 
export_overlaid_n_video(), respectively. In addition, this module also 
provides generic image and video export functions named, respectively, 
export_image() and export_video().

For storing motion data in external files, this module provides generic 
comma-separated-value (csv) file reader and writer named, respectively, 
read_csv() and write_csv().

'''

# import modules from python STL
import itertools, collections, csv
from types import SimpleNamespace

# import third-party modules
import numpy as np
import scipy, imageio, PIL, PIL.Image, PIL.ImageColor
import skimage, skimage.color, skimage.draw
import skimage.filters, skimage.measure, skimage.morphology
import sklearn, sklearn.cluster

# matplotlib magic and import

import matplotlib as mpl
import matplotlib.pyplot as plt

# convenient hooks

morph = skimage.morphology

skm = SimpleNamespace()
skm.color = skimage.color
skm.draw = skimage.draw
skm.filters = skimage.filters
skm.measure = skimage.measure
skm.morph = skimage.morphology

sp = SimpleNamespace()
sp.spat = scipy.spatial
sp.dist = scipy.spatial.distance
sp.opt = scipy.optimize

skl = SimpleNamespace()
skl.clust = sklearn.cluster

MotionSummary = collections.namedtuple("MotionSummary", 
    ["distance", "proportion"]
)

#### utility functions

def _peek_iterator(iterator):
    '''
    peek at the first yield of the iterator without "consuming" it
    
    ARGUMENTS:
      iterator: iterator that needs to be peeked at
      
    RETURNS:
      elem: the first yield from the iterator
      out_iterator: an iterator that function as a copy of unconsumed iterator
      
    NOTES:
      both elem and the first yield of out_iterator points to the same object
      which can have side effects if elem is mutable!
    '''
    iterator = iter(iterator)
    elem = next(iterator)
    out_iterator = itertools.chain([ elem ], iterator)
    
    return (elem, out_iterator)

def build_from_iter(
    iter_frames, step=1, start=0, stop=None, cropbox=None
):
    '''
    build a (3- or 4-axes) numpy array from a generator/iterator of 
    frames (2- or 3-axes numpy array)
    
    ARGUMENTS:
      iter_frames: an iterator yielding 2- or 3-axes numpy array, the zeroth
        index being row and first index being column
      step: steps between successive frames to be read
      start: the starting index of the frame to be read
      stop: the upper index (exclusive) of frame to be read (if None, 
        continues until the reader object reaches end)
      cropbox: the cropbox (left, top, right, bottom) to crop the image to
    
    RETURNS:
      an (n+1)-axes numpy array whose slices over the zeroth index are the
      frames from the iterators

    '''
    # set cropbox
    row_slice, col_slice = cropbox_to_slices(cropbox)
    
    # convert iterator to numpy array
    return np.array([ 
        __[row_slice, col_slice, ...] for __ in 
        itertools.islice(iter_frames, start, stop, step)
    ])


#### Image processing related codes

def build_grayscaled(
    frames, step=1, start=0, stop=None, cropbox=None, matrix=None
):
    '''
    build a SUBSAMPLE of frames from the video reader (or a 4-dim numpy array); 
    all frames read are assumed to be colored and are converted to grayscale 11
    (as uint8 numpy array)
    
    ARGUMENTS:
      frames: an iterator yielding image frames as 3-dim numpy array; or
        a 4-dim numpy array with shape ~(frame, row, col, RGB)
      start: the starting index of the frame to be read
      stop: the upper index (exclusive) of frame to be read (if None, 
        continues until the reader object reaches end)
      step: steps between successive frames to be read
      cropbox: the cropbox (left, top, right, bottom) to crop the image to
      matrix: 4-tuple of float, representing the matrix of RGB+const. to 
        luminosity conversion. If None uses PIL default
    
    RETURNS:
      a 3-dim numpy array, where 0th index corresponds to frame,
      1st index corresponds to row, and 2nd index corresponds to column
    
    NOTES:
      this function exits gracefully even if stop > # frames, in which case
      it simply gets to the end and returns
    '''
    if isinstance(frames, np.ndarray):
    
        # allocate output array and build it
        out = np.zeros_like(frames[start:stop:step,:,:,0], dtype=np.uint8)
        for i, raw in enumerate(frames[start:stop:step]):
            out[i] = PIL.Image.fromarray(
                raw
            ).crop(cropbox).convert("L", matrix=matrix)
        return out
        
    else:
        return np.array([
            np.array(PIL.Image.fromarray(
                raw
            ).crop(cropbox).convert("L", matrix=matrix)) for 
            raw in itertools.islice(frames, start, stop, step)
        ])

def iter_grayscaled(
    reader, step=1, start=0, stop=None, cropbox=None, matrix=None
):
    '''
    generator for iterating through grayscale-converted frames (as 2-dim 
    uint8 numpy array) from a SUBSAMPLE of frames from the video reader
    
    ARGUMENTS:
      reader: an imageio reader object representing colored frames of a 
        video or collated images
      step: steps between successive frames to be read
      start: the starting index of the frame to be read
      stop: the upper index (exclusive) of frame to be read (if None, 
        continues until the reader object reaches end)
      cropbox: the cropbox (left, top, right, bottom) to crop the image to
      matrix: 4-tuple of float, representing the matrix of RGB+const. to 
        luminosity conversion. If None uses PIL default
    
    RETURNS:
      an iterator of 2-dim numpy array, where 0th index corresponds to row, 
      and 2nd index corresponds to column
    
    NOTES:
      this function exits gracefully even if stop > # frames, in which case
      it simply gets to the end and returns
    '''
    for frame in itertools.islice(reader, start, stop, step):
        yield np.array(PIL.Image.fromarray(frame).crop(cropbox).convert(
            "L", matrix=matrix
        ))

def calc_background(
    frames, step=1, start=0, stop=None, cropbox=None, dtype="int16"
):
    '''
    compute a background frame by sub-sampling the given grayscaled 
    (uint8) frames
    
    ARGUMENTS:
      frames: an iterator of grayscaled frames (2-dim uint8 numpy arrays), 
        or a 3-dim uint numpy array
      step: steps between successive frames to be read
      start: the starting index of the frame to be read
      stop: the upper index (exclusive) of frame to be read (if None, 
        continues until the reader object reaches end)
      cropbox: the cropbox (left, top, right, bottom) to crop the image to
      dtype: the dtype for which the pixel of the background frame would be
        cast into
    
    RETURNS:
      a 2-dim numpy array with the same shape as input frames
    
    NOTES:
      the background is computed by taking the MEDIAN of the sub-sampled frames
    '''
    # set cropbox
    row_slice, col_slice = cropbox_to_slices(cropbox)
    
    # compute background frame
    if isinstance(frames, np.ndarray):
        return np.median(
            frames[start:stop:step, row_slice, col_slice], axis=0
        ).astype(dtype)
    else:
        return np.median([
            __[row_slice, col_slice] for __ in 
            itertools.islice(frames, start, stop, step)
        ], axis=0).astype(dtype)

def build_subtracted(
    frames, background, step=1, start=0, stop=None, cropbox=None, 
    sample_step=10
):
    '''
    build a SUBSAMPLE of subtracted frames from the given iterator of frames 
    (2-dim uint8 numpy array) or 3-dim uint8 numpy array
    
    ARGUMENTS:
      frames: an iterator yielding grayscaled frames as 2-dim numpy array; 
        or a 3-dim numpy array with shape ~ (frame, row, col)
      background: the background image to be subtracted from (MUST be of
        signed dtype), or None if to be calculated automatically
      step: steps between successive frames to be read
      start: the starting index of the frame to be read
      stop: the upper index (exclusive) of frame to be read (if None, 
        continues until the reader object reaches end)
      cropbox: the cropbox (left, top, right, bottom) to crop the image to
      sample_step: steps between successive SUBSAMPLED frame to be read
        for the purpose of calculating background. No effects if background 
        is supplied
    
    RETURNS:
      a 3-dim numpy array, where 0th index corresponds to frame,
      1st index corresponds to row, and 2nd index corresponds to column
      if background is computed automatically it is also returned
    
    NOTES:
      1/ background could have dtype=float, even though output has dtype=uint8
      2/ the background image must either have the exact shape as the 
        CROPPED frame, or must encompass the entire range of cropbox (if 
        cropbox is None it must have the same shape as the frames)
    '''
    # set cropbox
    row_slice, col_slice = cropbox_to_slices(cropbox)
    
    # build output frame
    if isinstance(frames, np.ndarray):
        out_frames = frames[start:stop:step, row_slice, col_slice].copy()
    else:
        out_frames = np.array([
            __[row_slice, col_slice] for __ in 
            itertools.islice(frames, start, stop, step)
        ])
    
    if background is None:
        # compute background frame if none supplied
        bg = calc_background(out_frames, sample_step)
    else:
        # otherwise use the supplied background, crop if needed
        bg = background if ( (cropbox is None) or (background.shape == (
            cropbox[3] - cropbox[1], cropbox[2] - cropbox[0]
        )) ) else background[row_slice, col_slice]
    
    # compute and return subtracted frame
    for i, frame in enumerate(out_frames):
        out_frames[i] = np.abs(frame - bg).astype(np.uint8)
    
    if background is None:
        return (out_frames, bg)
    else:
        return out_frames

def iter_subtracted(
    reader, background, step=1, start=0, stop=None, 
    cropbox=None, matrix=None
):
    '''
    generator for iterating through grayscale-converted and background-
    subtracted frames (as 2-dim uint8 numpy array) from a SUBSAMPLE of 
    frames from the video reader
    
    ARGUMENTS:
      reader: an imageio reader object representing colored frames of a 
        video or collated images
      background: the background image to be subtracted from (MUST be of
        signed dtype)
      step: steps between successive frames to be read
      start: the starting index of the frame to be read
      stop: the upper index (exclusive) of frame to be read (if None, 
        continues until the reader object reaches end)
      cropbox: the cropbox (left, top, right, bottom) to crop the image to
      matrix: 4-tuple of float, representing the matrix of RGB+const. to 
        luminosity conversion. If None uses PIL default
    
    RETURNS:
      an iterator of 2-dim numpy array, where 0th index corresponds to row, 
      and 2nd index corresponds to column
    
    NOTES:
      1/ background could have dtype=float, even though output has dtype=uint8
      2/ the background image must either have the exact shape as the 
        CROPPED frame, or must encompass the entire range of cropbox (if 
        cropbox is None it must have the same shape as the frames)
    '''
    # crop background image if needed
    bg = background if ( (cropbox is None) or (
        background.shape == (cropbox[3] - cropbox[1], cropbox[2] - cropbox[0])
    ) ) else background[cropbox_to_slices(cropbox)]
    
    for frame in iter_grayscaled(reader, step, start, stop, cropbox, matrix):
        yield np.abs(frame - bg).astype(np.uint8)

def count_intensity(frames, step=1, start=0, stop=None, cropbox=None):
    '''
    count the frequency of each luminosity values (from 0 to 255) in a 
    sub-sample of grayscale images frames
    
    ARGUMENTS:
      frames: an iterator of grayscaled frames (2-dim uint8 numpy arrays), 
        or a 3-dim uint numpy array
      step: steps between successive frames to be read
      start: the starting index of the frame to be read
      stop: the upper index (exclusive) of frame to be read (if None, 
        continues until the reader object reaches end)
      cropbox: the cropbox (left, top, right, bottom) to crop the image to
      
    RETURNS:
      a 1-dim (len=256) numpy array
    '''
    row_slice, col_slice = cropbox_to_slices(cropbox)
    
    if isinstance(frames, np.ndarray):
        counts = np.bincount(
            frames[start:stop:step, row_slice, col_slice].ravel(), 
            minlength=256
        )
    else:
        counts = np.zeros((256,))
        for fr in itertools.islice(frames, start, stop, step):
            counts += np.bincount(
                fr[row_slice, col_slice].ravel(), minlength=256
            )[:256]
    
    return counts / np.sum(counts)

def calc_threshold(frames, step=1, start=0, stop=None, cropbox=None):
    '''
    compute binary threshold by sub-sampling the given grayscaled (uint8) 
    frames
    
    ARGUMENTS:
      frames: an iterator of grayscaled frames (2-dim uint8 numpy arrays), 
        or a 3-dim uint numpy array
      step: steps between successive frames to be read
      start: the starting index of the frame to be read
      stop: the upper index (exclusive) of frame to be read (if None, 
        continues until the reader object reaches end)
      cropbox: the cropbox (left, top, right, bottom) to crop the image to
    
    RETURNS:
      the value of threshold
    '''
    # set cropbox
    row_slice, col_slice = cropbox_to_slices(cropbox)
    
    # compute threshold
    if isinstance(frames, np.ndarray):
        return skm.filters.threshold_otsu(
            frames[start:stop:step, row_slice, col_slice]
        )
    else:
        return skm.filters.threshold_otsu(np.array([
            __[row_slice, col_slice] for __ in 
            itertools.islice(frames, start, stop, step)
        ]))

def build_thresholded(
    frames, threshold=None, step=1, start=0, stop=None, 
    cropbox=None, mask=None, sample_step=10
):
    '''
    build a SUBSAMPLE of thresholded binary frames from the given iterator of 
    frames (2-dim uint8 numpy array) or 3-dim uint8 numpy array
    
    ARGUMENTS:
      frames: an iterator yielding grayscaled frames as 2-dim numpy array; 
        or a 3-dim numpy array with shape ~ (frame, row, col)
      threshold: the value of threshold to be used (if None, the threshold 
        is determined automatically)
      step: steps between successive frames to be read
      start: the starting index of the frame to be read
      stop: the upper index (exclusive) of frame to be read (if None, 
        continues until the reader object reaches end)
      cropbox: the cropbox (left, top, right, bottom) to crop the image to
      mask: an additional mask that further clips the resulting binary images
      sample_step: steps between successive SUBSAMPLED frame to be read
        for the purpose of calculating background. No effects if threshold 
        is supplied
    
    RETURNS:
      a 3-dim bool numpy array with the same dimensions as the given frames;
      if the threshold is computed automatically it is also returned
    
    NOTE:
      the mask is applied at the END of the process, and thus, has no effects
      on the value of threshold. In contrast, the cropbox is applied BEFORE
      the threshold is computed if it is to be computed automatically
    '''
    # set cropbox
    row_slice, col_slice = cropbox_to_slices(cropbox)
    
    # compute threshold if needed
    thres = calc_threshold(
        frames, sample_step * step, start, stop, cropbox
    ) if (threshold is None) else threshold
    
    if isinstance(frames, np.ndarray):
        if mask is None:
            out_frames = np.array(
                frames[start:stop:step, row_slice, col_slice] > thres
            )
        else:
            out_frames = np.logical_and(
                frames[start:stop:step, row_slice, col_slice] > thres, mask
            )
    else:
        if mask is None:
            out_frames = np.array([ 
                __[row_slice, col_slice] > thres for __ in 
                itertools.islice(frames, start, stop, step)
            ])
        else:
            out_frames = np.array([ 
                np.logical_and(__[row_slice, col_slice] > thres, mask) for
                __ in itertools.islice(frames, start, stop, step)
            ])
    
    # return the proper results
    if threshold is None:
        return (out_frames, thres)
    else:
        return out_frames

def iter_thresholded(
    reader, background, threshold, step=1, start=0, stop=None, cropbox=None, 
    mask=None, matrix=None
):
    '''
    generator for iterating through grayscale-converted, background-
    subtracted, and binary thresholded frames (as 2-dim bool numpy array) from
    a SUBSAMPLE of frames from the video reader
    
    ARGUMENTS:
      reader: an imageio reader object representing colored frames of a 
        video or collated images
      background: the background image to be subtracted from (MUST be of
        signed dtype)
      threshold: the value of threshold to be used
      step: steps between successive frames to be read
      start: the starting index of the frame to be read
      stop: the upper index (exclusive) of frame to be read (if None, 
        continues until the reader object reaches end)
      cropbox: the cropbox (left, top, right, bottom) to crop the image to
      mask: an additional mask that further clips the resulting binary images
      matrix: 4-tuple of float, representing the matrix of RGB+const. to 
        luminosity conversion. If None uses PIL default
    
    RETURNS:
      an iterator of 2-dim numpy array, where 0th index corresponds to row, 
      and 2nd index corresponds to column
      if the threshold is computed automatically it is also returned
    
    NOTES:
      1/ the mask is applied at the END of the process, and thus, has no 
        effects on the value of threshold. In contrast, the cropbox is applied 
        BEFORE the threshold is computed if it is to be computed automatically
      2/ the background image must either have the exact shape as the 
        CROPPED frame, or must encompass the entire range of cropbox (if 
        cropbox is None it must have the same shape as the frames)
    '''
    if mask is None:
        for frame in iter_subtracted(
            reader, background, step, start, stop, cropbox, matrix
        ):
            yield np.array(frame > threshold)
        
    else:
        for frame in iter_subtracted(
            reader, background, step, start, stop, cropbox, matrix
        ):
            yield np.logical_and(frame > threshold, mask)

def build_postprocessed(
    frames, Ops, params=None, kwparams=None, *, in_place=False
):
    '''
    apply a sequence post-processing operations on the given frames
    
    ARGUMENTS:
      frames: a 3-dim numpy array with shape ~ (frame, row, col); or an
        iterator yielding 2-dim numpy array with shape ~ (row, col)
      Ops: a sequence of functions, each operate on 2-dim (3 with colorspace?)
        memory view of numpy array and return an array of same shape
      params: list of list, with the i-th element corresponds to positional 
        arguments to be passed to the i-th operation
      kwparams: list of dict, with the i-th element corresponds to keyword 
        arguments to be passed to the i-th operation
      in_place: if true, the frames are modified in-place. Ignored (treated
        as false) if frames is an iterator
    
    RETURNS:
      if in_place is True, a 3-dim (4 with colorspace?) numpy array with the 
      same dimensions as the given frames. Otherwise None
    
    NOTES:
      many useful post-processing functions can be found inside the 
      skimage.morphology module, which is mapped to the "morph" variable
      of this package for your convenience
    '''
    
    # deal with the special case of no parameters
    if params is None:
        params = []
        params = [ params for __ in Ops ]
    
    # deal with the special case of no keyword parameters
    if kwparams is None:
        kwparams = dict()
        kwparams = [ kwparams for __ in Ops ]
    
    if isinstance(frames, np.ndarray):
        if in_place:
            new_frames = frames
        else:
            new_frames = np.zeros_like(frames)
    else: # construct the array and then manipulate in place
        in_place = False
        frames = np.array([ __ for __ in frames ])
        new_frames = frames
        
    for i, frame in enumerate(frames):
        for op, args, kwargs in zip(Ops, params, kwparams):
            frame = op(frame, *args, **kwargs)
        new_frames[i,:,:] = frame

    if in_place:
        return None
    else:
        return new_frames

def iter_postprocessed(reader, background, threshold, Ops, 
    params=None, kwparams=None, step=1, start=0, stop=None, 
    cropbox=None, mask=None, matrix=None
):
    '''
    generator to yield post-processed frames from imageio reader object, where
    the post-processing is done after the frame is grayscale-converted, 
    background-subtracted, and binary-thresholded.
    
    ARGUMENTS:
      reader: an imageio reader object representing colored frames of a 
        video or collated images
      background: the background image to be subtracted from (MUST be of
        signed dtype)
      threshold: the value of threshold to be used
      Ops: a sequence of functions, each operate on 2-dim (3 with colorspace?)
        memory view of numpy array and return an array of same shape
      params: list of list, with the i-th element corresponds to positional 
        arguments to be passed to the i-th operation
      kwparams: list of dict, with the i-th element corresponds to keyword 
        arguments to be passed to the i-th operation
      step: steps between successive frames to be read
      start: the starting index of the frame to be read
      stop: the upper index (exclusive) of frame to be read (if None, 
        continues until the reader object reaches end)
      cropbox: the cropbox (left, top, right, bottom) to crop the image to
      mask: an additional mask that further clips the resulting binary images
      matrix: 4-tuple of float, representing the matrix of RGB+const. to 
        luminosity conversion. If None uses PIL default
    
    RETURNS:
      an iterator of 2-dim numpy array, where 0th index corresponds to row, 
      and 2nd index corresponds to column
    
    NOTES:
      1/ many useful post-processing functions can be found inside the 
        skimage.morphology module, which is mapped to the "morph" variable
        of this package for your convenience
      2/ the mask is applied at the END of the process, and thus, has no 
        effects on the value of threshold. In contrast, the cropbox is applied 
        BEFORE the threshold is computed if it is to be computed automatically
      3/ the background image must either have the exact shape as the 
        CROPPED frame, or must encompass the entire range of cropbox (if 
        cropbox is None it must have the same shape as the frames)
    '''
    
    # deal with the special case of no parameters
    if params is None:
        params = []
        params = [ params for __ in Ops ]
    
    # deal with the special case of no keyword parameters
    if kwparams is None:
        kwparams = dict()
        kwparams = [ kwparams for __ in Ops ]
    
    for frame in iter_thresholded(
        reader, background, threshold, step, start, stop, cropbox, mask, matrix
    ):
        for op, args, kwargs in zip(Ops, params, kwparams):
            frame = op(frame, *args, **kwargs)
        yield frame

def iter_postprocess(iterator, Ops, params=None, kwparams=None):
    '''
    generator to yield post-processed frames from pre-processed ones
    
    ARGUMENTS:
      iterator: an iterator of 2-dim numpy array of shape ~ (row, col)
      Ops: a sequence of functions, each operate on 2-dim (3 with colorspace?)
        memory view of numpy array and return an array of same shape
    
    RETURNS:
      an iterator over the frames contained in the original iterator
    
    NOTES:
      many useful post-pocessing functions can be found inside the 
      skimage.morphology module, which is mapped to the "morph" variable
      of this package for your convenience
    '''
    
    # deal with the special case of no parameters
    if params is None:
        params = []
        params = [ params for __ in Ops ]
    
    # deal with the special case of no keyword parameters
    if kwparams is None:
        kwparams = dict()
        kwparams = [ kwparams for __ in Ops ]
    
    for frame in iterator:
        for op, args, kwargs in zip(Ops, params, kwparams):
            frame = op(frame, *args, **kwargs)
        yield frame

#### Image region/path identification related codes

def mark_image(
    image, num, convex=True, max_tries=3, timeout=0, title="", *, 
    backend="qt5agg", cmap="gray", size=(16,10), fontsizes=(20,14), 
    raise_error=True, message=(
        "Left click to select, right click to undo, middle click to" +
        " terminate/retry.\n One final left click after" +
        " all points are selected to return [try # {} out of {}]"
    )
):
    '''
    record specific pixel coordinates from an image using an interactive prompt
    
    ARGUMENTS:
      image: the (grayscale) image from which pixel coordinates are to be 
        extracted, represented as a 2-dim uint8 numpy array
      num: the number of pixel coordinates (x_i, y_i) to be extracted
      convex: if True and num > 2, the pixel coordinates are understood to 
        define a convex hull. Thus, only the bounding vertices are returned 
        and these are rearranged in uni-clockwise order
      max_tries: the number of tries for proper import before the function
        gives up and returns / raises error
      title: title to be displayed on top of image in interactive prompt
      backend: the backend used by matplotlib interactive prompt (ginput)
      cmap: the color map used by matplotlib in displaying the image; must
        be a valid specification to make_cmap()
      size: size (width, height) of the image in the interactive prompt
      fontsize: font-size of the (0th index) title and the (1st index) message
      timeout: timeout until which the prompt is automatically close. If <= 0
        the prompt will not close except manually
      raise_error: if True, raise ValueError if wrong number of points are
        received. Otherwise return whatever is received.
      message: message to display below the title (NOTE: message is passed
        the current try number and max_tries as .format() arguments)
        
    RETURNS:
      list of tuples of pixel coordinates
      
    NOTES:
      the convex option is helpful if the points represents a convex region, 
      since with convex=True the order of which the points are selected becomes
      immaterial
    '''
    
    # create actual cmap from specification
    cmap = make_cmap(cmap)
    
    # NOTE: for better interface, secretly asked for (num + 1) points, 
    # but throw out the coordinates of the last input
    real_num = num + 1
    for i in range(1, max_tries + 1):
        mpl.use(backend)
        plt.figure(figsize=size)
        plt.imshow(image, cmap=cmap)
        plt.suptitle(title, fontsize=fontsizes[0]) 
        plt.title(message.format(i, max_tries), fontsize=fontsizes[1])
        points = plt.ginput(real_num, timeout=timeout)
        plt.close()
        if len(points)==real_num:
            points = points[:-1]
            break
    else: # else of for => no break
        if raise_error:
            raise ValueError("Unexpected number of pixel coordinates")
    
    # filter and rearrange captured pixel coordinates if they 
    # represent a convex hull
    if convex and (len(points) > 2):
        hull = sp.spat.ConvexHull(points)
        points = [ points[__] for __ in hull.vertices ]
        
    return points

def vertices_to_cropbox(points, block_size=1):
    '''
    compute the minimal cropbox encompassing the given set of vertices in 2D
    
    ARGUMENTS:
      points: vertices (x_i, y_i) defining the bounded region, as a sequence 
        of sequences (e.g., list of tuples)
      block_size: an integer for which the corners are required to be
        integer multiples of. Ignored if None
        
    RETURNS:
      a 4-tuple giving a PIL-style cropbox (left, top, right, bottom)
    '''
    
    pt_min = np.floor(np.min(points,axis=0)).astype(int)
    pt_max = np.ceil(np.max(points,axis=0)).astype(int)
    
    if block_size is not None:
        block_size = int(block_size) # defensive
        pt_min = (pt_min // block_size) * block_size
        q, r = np.divmod(pt_max, block_size)
        pt_max = (q + np.sign(r)) * block_size
    
    return tuple(np.hstack([pt_min, pt_max]))

def cropbox_to_slices(cropbox):
    '''
    convert a PIL-style cropbox (left, top, right, bottom) into the 
    corresponding numpy index slice
    
    ARGUMENTS:
      cropbox: a PIL-style cropbox, which can be None
        
    RETURNS:
      a 2-tuple of python slice object, which can be used to subset 2-dim 
      numpy arrays
    
    NOTE that in the returned slice, the 0th dimension corresponds to the 
    y coordinate of image while the 1st dimension corresponds to the 
    x coordinate
    '''
    if cropbox is None:
        return (slice(None), slice(None))
    else:
        return np.index_exp[cropbox[1]:cropbox[3], cropbox[0]:cropbox[2]]

def cropbox_to_vertices(cropbox, px2real=1.0):
    '''
    convert a PIL-style cropbox in (left, top, right, bottom) pixel 
    coordinates to vertices of the corresponding rectangle

    ARGUMENTS:
      cropbox: a PIL-style cropbox
      px2real: the conversion scale from pixel to physical unit. i.e., the 
        physical length that 1 pixel corresponds to
        
    RETURNS:
      a len=4 python list for which each element is a 2-tuple. Each 2-tuple
      corresponds to the (x, y) coordinates of a vertex, and the vertices 
      tranverse counterclockwise from (left, top)
    '''
    
    cropbox = [ __ * px2real for __ in cropbox]
    
    return [
        (cropbox[0], cropbox[1]), (cropbox[0], cropbox[3]), 
        (cropbox[2], cropbox[3]), (cropbox[2], cropbox[1])
    ]

def vertices_to_mask(points, *, image=None, cropbox=None):
    '''
    create an image mask representing the region bounded by the given 
    pixel coordinates
    
    ARGUMENTS:
      points: vertices (x_i, y_i) defining the bounded region in pixel 
        coordinates, as a sequence of sequences (e.g., list of tuples)
      image: a 2-dim numpy array (3-dim with colorspace) representing the 
        image inside which the vertices are defined, assumed to enclose 
        all vertices; Ignored and can be None if cropbox is specified
      cropbox: sequence giving a PIL-style cropbox (left, top, right, bottom);
        can be None if image is specified. If not None, the mask is cropped to
        the cropbox
    
    RETURN:
      a 2-dim bool numpy array of the same dimension of the cropped image
      
    NOTES:
      the order of vertices matter! e.g., 4 points may define a trapezium 
      or a "bow-tie" depending on order
    '''
    
    if image is None and cropbox is None:
        raise ValueError("image can cropbox cannot both be None")
    
    if cropbox is None:
        offset = (0, 0)
        shape = image.shape[:2]
    else:
        offset = (cropbox[0], cropbox[1])
        shape = (cropbox[3] - cropbox[1], cropbox[2] - cropbox[0])
    
    # NOTE that skimage expects coordinates in row#, col# form,
    # which is OPPOSITE to the usual (x, y) form
    row_col = np.array([ 
        (__[1] - offset[1], __[0] - offset[0]) for __ in points 
    ])
    return skm.draw.polygon2mask(shape, row_col)

def vertices_to_path(points, scale=1, offset=(0,0), closed=True):
    '''
    create an matplotlib path object representing the boundary obtained
    by joining the given pixel coordinates
    
    ARGUMENTS:
      points: vertices (x_i, y_i) in pixel coordinates defining the boundary, 
        as a sequence of sequences (e.g., list of tuples)
      scale: a scaling factor to be applied to all coordinates, intended for 
        converting between pixel and "real" coordinates
      offset: a 2-tuple (x, y) of offset value in pixel coordinates (e.g., 
        the first two component of cropbox), to be subtracted from points
        BEFORE scale is applied
      closed: whether the vertices represent a closed path
    
    RETURN:
      a matplotlib path object
      
    NOTES:
      1/ the order of vertices matter! e.g., 4 points may define a trapezium 
        or a "bow-tie" depending on order
      2/ offset can if fact contain more component. In particular, it can be
        a PIL-style cropbox
    '''
    
    points = [ scale * (np.array(__) - offset) for __ in points ]
    
    if closed:
        return mpl.path.Path(points + [ points[0] ], closed=True)
    else:
        return mpl.path.Path(points, closed=False)

def calc_pixel_to_phys(coords_px, len_phys):
    '''
    compute the pixel-to-physical unit conversion factor
    
    ARGUMENTS:
      coords_px: a pair (x_i, y_i) of pixel coordinates of known length as a 
        sequence of sequences (e.g., list of tuples)
      len_phys: distance between the two coordinates in physical units

    RETURNS:
      the pixel-to-physical unit conversion factor
    '''
    
    len_px = sp.dist.euclidean(coords_px[0], coords_px[1])
    return len_phys / len_px

def convert_to_physical(points, px2real=1.0, offset=(0,0)):
    '''
    Convert data points from pixel coordinates to physical coordinates
    
    ARGUMENTS:
      points: array-like or matplotlib path or None; if array-like the 
        innermost (last) index must have length 2, if matplotlib path must 
        be a path in 2D
      px2real: the conversion scale from pixel to physical unit. i.e., the 
        physical length that 1 pixel corresponds to
      offset: the offset of the origin of the physical coordinates relative
        to the pixel coordinates, in *pixel* unit, in (x,y) convention. None
        is treated as an alias of (0, 0)
    
    RETURNS:
      if points is array-like: a numpy array of the same shape as the 
      (converted, if applicable) input; 
      if points is matplotlib path: a corresponding path
      if points is None: the matplotlib transform object
    
    NOTES: offset can contains more than 2 components. In particular, it can
    be a PIL-style cropbox.
    '''
    if offset is None:
        offset = (0, 0)
    
    converter = mpl.transforms.Affine2D()
    converter = converter.translate(-offset[0], -offset[1]).scale(px2real)
    
    if points is None:
        return converter
    
    if isinstance(points, mpl.path.Path):
        return converter.transform_path(points)
    
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    old_shape = points.shape
    return converter.transform(points.reshape(-1, 2)).reshape(old_shape)

def convert_to_pixel(points, px2real=1.0, offset=(0, 0), quantize=True):
    '''
    Convert data points from physical coordinates to pixel coordinates
    
    ARGUMENTS:
      points: array-like or matplotlib path or None; if array-like the 
        innermost (last) index must have length 2, if matplotlib path must 
        be a path in 2D
      px2real: the conversion scale from pixel to physical unit. i.e., the 
        physical length that 1 pixel corresponds to
      offset: the offset of the origin of the physical coordinates relative
        to the pixel coordinates, in *pixel* unit, in (x,y) convention. None
        is treated as an alias of (0, 0)
      quantize: if True, the coordinates are casted to integer before 
        returning; ignored if points is None
    
    RETURNS:
      if points is array-like: a numpy array of the same shape as the 
      (converted, if applicable) input; 
      if points is matplotlib path: a corresponding path
      if points is None: the matplotlib transform object
    
    NOTES: 
      1/ offset can contains more than 2 components. In particular, it can
        be a PIL-style cropbox.
      2/ for matplotlib path the quantization relies on the path's own 
        .cleaned() method
    '''
    if offset is None:
        offset = (0, 0)
    
    converter = mpl.transforms.Affine2D()
    converter = converter.scale(1/px2real).translate(offset[0], offset[1])
    
    if points is None:
        return converter
        
    if isinstance(points, mpl.path.Path):
        if quantize:
            return converter.transform_path(points).cleaned(snap=True)
        else:
            return converter.transform_path(points)
    
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    
    old_shape = points.shape
    points = converter.transform(points.reshape(-1, 2)).reshape(old_shape)
    if quantize:
        return np.rint(points).astype(int) # round and cast to int
    else:
        return points

#### Analysis related codes

def compute_centroids(frames):
    '''
    Compute the (intensity) centroid from the given frames
        
    ARGUMENTS:
      frames: 3-dim numpy array representing a sequence of images, with the 
        0th index being the sequence index; or an iterator yielding successive
        images as 2-dim numpy arrays
    
    RETURNS:
      a sequence of (x_i, y_i) coordinates as 2-dim numpy array, with the
        0th index being the sequence index
    '''
    if isinstance(frames, np.ndarray):
        
        outarray = np.zeros( (frames.shape[0], 2), dtype=float )
        
        for i, frame in enumerate(frames):
            
            M = skm.measure.moments(frame, order=1)
            # x (horizontal, count from left)
            outarray[i,0] = M[0,1] / M[0,0] 
            # y (vertical, count from top)
            outarray[i,1] = M[1,0] / M[0,0] 
    
    else:
        
        outarray = np.zeros( (0,), dtype=float )
        
        for frame in frames:
            
            M = skm.measure.moments(frame, order=1)
            outarray = np.append(outarray, [
                M[0,1] / M[0,0],
                M[1,0] / M[0,0],
            ])
        
        # reshape result
        outarray = np.reshape(outarray, (-1, 2))
    
    return outarray

def compute_n_centroids(
    frames, n, *, reinitialize=False, **kwargs
):
    '''
    Compute the centroid from the given frames, assuming there are exactly 
    n objects
        
    ARGUMENTS:
      frames: 3-axes numpy array representing a sequence of images, with the 
        0th index being the sequence index; or an iterator yielding successive
        images as 2-axes numpy arrays
      n: the number of objects in the frames
      chain_init: whether to randomize the initial guess for centroids for each
        frame (if True), or to just use the centroid from the previous frame
        as initial guess
      (**kwargs): remaining keyword arguments are passed to the 
        MiniBatchKMeans engine used to identify objects
      
    RETURNS:
      a sequence of (x_i, y_i) coordinates as 3-axes numpy array, with the
        0th axis being the sequence index, the 1st axis being the object 
        index, and the 2nd axis being the (x,y) index
    '''
    
    def recall(X, n_clust, random_state):
        '''
        recall the last computed centroids
        '''
        try:
            return old_coords
        except Exception as e:
            return skl.clust.kmeans_plusplus(
                X, n_clust, random_state=random_state
            )
    
    # initialize k-mean engine
    if reinitialize:
        engine = skl.clust.MiniBatchKMeans(n, compute_labels=False, **kwargs)
    else:
        engine = skl.clust.MiniBatchKMeans(
            n, init=recall, compute_labels=False, **kwargs
        )
    
    # main loop
    if isinstance(frames, np.ndarray):
        
        # initialize output, initialize coordinates for matching
        outarray = np.zeros( (frames.shape[0], n, 2), dtype=float )
        
        if reinitialize:
            old_coords = np.zeros((n, 2), dtype=float)
        else:
            features = np.vstack(frames[0].nonzero()).T
            old_coords, *_ = skl.clust.kmeans_plusplus(features, n)
        
        for i, frame in enumerate(frames):
            
            # array of (row, col) of "bright" pixels
            features = np.vstack(frame.nonzero()).T
            
            # find the centroids via k-mean clustering
            engine.fit(features)
            new_coords = engine.cluster_centers_
            
            # match coordinates from previous frame to current frame, 
            # assuming that pairwise same-label distance is to be minimized
            dist_pairs = sp.dist.cdist(old_coords, new_coords)
            assignment = sp.opt.linear_sum_assignment(dist_pairs)
            
            # re-label centroids by matched label, and cache for next frame
            old_coords = new_coords[assignment[1], :]
            
            # convert (row, col) pixel coordinates to (x, y) real coordinates
            # append result to out_array
            outarray[i,:,:] = old_coords[:,::-1]
            
    else:
        
        # initialize output, initialize coordinates for matching
        outarray = np.zeros( (0,), dtype=float )
        
        # initialize guess
        if reinitialize:
            old_coords = np.zeros((n, 2), dtype=float)
        else:
            init_frame, frames = _peek_iterator(frames)
            features = np.vstack(init_frame.nonzero()).T
            old_coords, *_ = skl.clust.kmeans_plusplus(features, n)
        
        for i, frame in enumerate(frames):
        
            # array of (row, col) of "bright" pixels
            features = np.vstack(frame.nonzero()).T
            
            # find the centroids via k-mean clustering
            engine.fit(features)
            new_coords = engine.cluster_centers_
            
            # match coordinates from previous frame to current frame, 
            # assuming that pairwise same-label distance is to be minimized
            dist_pairs = sp.dist.cdist(old_coords, new_coords)
            assignment = sp.opt.linear_sum_assignment(dist_pairs)
            
            # re-label centroids by matched label, cache result for next frame
            old_coords = new_coords[assignment[1], :]
            
            # convert (row, col) pixel coordinates to (x, y) real coordinates
            # append result to outarray
            outarray = np.append(outarray, old_coords[:,::-1])
        
        # reshape result
        outarray = np.reshape(outarray, (-1, n, 2))
    
    return outarray

def calculate_motion(
    coords, px2real=1.0, dt=1, *, offset=(0,0), incl_speed=True
):
    '''
    Compute velocity vector (and possibly speed) from coordinates
    
    ARGUMENTS:
      coords: (2+n)-dim numpy array (n >=0) representing a sequence of 
        coordinates, with the 0th index being the temporal index, and the last
        index being the spatial-component (i.e., (x, y)) index
      dt: time difference between successive coordinates
      incl_speed: whether the speed is also computed
      
    RETURNS:
      a (2+n)-dim numpy array (n >=0) representing velocities, with the last
      index being velocity components (i.e., (v_x, v_y), or (v_x, v_y, v) 
      if incl_speed is True) index, and all previous indices in agreement with
      that of the input coords, except that the 0-th (temporal) index is 
      one element shorter
    '''
    # translate offset
    if offset is None:
        offset = (0, 0)
    
    coords = (coords - np.array(offset)) * px2real
    
    vels = np.diff(coords, axis=0) / dt
    if incl_speed:
        speeds = np.linalg.norm(vels, axis=-1)
        out_array = np.concatenate(
            (coords, vels, np.expand_dims(speeds, -1)), axis=-1
        )
    return out_array

def estimate_motion(
    coords, px2real=1.0, dt=1.0, window=5, pad_value=np.nan, *, 
    rm_nan=True, offset=(0,0), t0=0.0, incl_speed=True, rcond=None
):
    '''
    Estimate positions and velocities of object(s) from raw physical 
    coordinates via rolling simple linear least-square fit
    
    ARGUMENTS:
      coords: (1+n+1)-axes numpy array (n >=0) representing a sequence of 
        coordinates, with the 0th index being the temporal index, and the last
        index being the spatial-component (i.e., (x, y)) index
      px2real: the conversion scale from pixel to physical unit. i.e., the 
        physical length that 1 pixel corresponds to
      dt: time difference between successive coordinates
      window: the window for computing rolling average and rolling slope. 
        Must be an odd integer
      pad_value: the value to pad the first and last window//2 entries of
        the output array
      rm_nan: whether nan values should be ignored in local least square fit
        (if True) or propagated (if False)
      offset: the offset of the origin of the physical coordinates relative
        to the pixel coordinates, in *pixel* unit, in (x,y) convention. None
        is treated as an alias of (0, 0)
      t0: the initial time instant
      incl_speed: whether the speed is also computed
      rcond: the rcond parameter passed to numpy.linalg.lstsq
    
    RETURNS:
      a (1+n+1)-axes numpy array (n >=0) representing motion data, with the 
      last index being (t, x, y, v_x, v_y), [or (t, x, y, v_x, v_y, v) if 
      incl_speed is True], where:
        t: the time point value, starting from 0 and construct from dt
        x: x- (horizontal) position of object, constructed via rolling average
        y: y- (vertical) position of object, constructed via rolling average
        v_x: x-component of velocity, constructed via rolling slope
        v_y: y-component of velocity, constructed via rolling slope
        v: speed of object, computed from v_x and v_y
      all other indices of the output are in agreement and align with that
      that of the input coords. To do so, the first and last (window // 2) 
      entries in the outermost index of the array (except the t-component)
      are set to numpy.nan
    '''
    # translate offset
    if offset is None:
        offset = (0, 0)

    # enforce window being an odd integer
    if (window % 2 != 1):
        raise ValueError("window must be an odd integer")
    shift = window // 2

    # reshape coords to unify 2-dim and (2+n)-dim cases
    old_shape = coords.shape
    coords = coords.reshape(old_shape[0], -1, old_shape[-1])

    # allocate output array (reshape later)
    inner_shape = 6 if incl_speed else 5
    out_array = np.full((*coords.shape[:-1], inner_shape), pad_value)

    # the design matrix, common across all cases
    Z = np.vstack((np.ones((window,)), (np.arange(window) - shift) * dt)).T

    # default indices to take in linear least square (when no nan)
    x_take = slice(window)
    y_take = slice(window)

    for i in range(coords.shape[1]):

        out_array[:,i,0] = np.arange(coords.shape[0]) * dt + t0

        if rm_nan:
            x_take = ~np.isnan(coords[j:j+window, i, 0])
            y_take = ~np.isnan(coords[j:j+window, i, 1])

        for j in range(coords.shape[0] - window + 1):
            out_array[j+shift,i,1:5:2], *_ = np.linalg.lstsq(
                Z[xtake,:], 
                px2real * (coords[j:j+window, i, 0][xtake] - offset[0]),
                rcond=rcond
            )
            out_array[j+shift,i,2:5:2], *_ = np.linalg.lstsq(
                Z[ytake,:], 
                px2real * (coords[j:j+window, i, 1][ytake] - offset[1]),
                rcond=rcond
            )

    if incl_speed:
        speeds = np.linalg.norm(out_array[shift:-shift,:,3:5], axis=-1)
        out_array[shift:-shift,:,5] = speeds

    # reshape to match input
    out_array = out_array.reshape((*old_shape[:-1], inner_shape))

    return out_array

def assemble_motion_data(
    coords, dt=1, marked_region=None, pad_value=np.nan, smooth=True, *, 
    window=5, t0=0.0, incl_speed=True, summary=True
):
    '''
    Compute all motion-related data from given coordinates
    
    ARGUMENTS:
      coords: 2- or 3-dim numpy array representing a time-series of coordinates
        of n objects, with the 0th index being the time-series index, the
        1st index being the object index, and the 2nd index being the spatial
        axes (x, y) index
      dt: time difference between successive coordinates
      marked_region: a region of interest (ROI) for which occupation is 
        checked if None assumes no such region
      pad_value: the value to pad motion data that cannot be determined
      smooth: whether to smoothen the raw coordinates in computing the motion
      window: the size of (rectangular) rolling window used to compute the
        smoothen motion. Relevant only if smooth is True
      t0: the initial time instant
      incl_speed: whether the speed is also computed
      summary: whether summary data is also returned
      
    RETURNS:
      a 2-tuple if summary is False (a 3-tuple otherwise), consisting of:
      1/ a 2- or 3-dim numpy array, where indices except last align with
        that of the input coords array, and the last index consist of motion
        data in the following order:
        t: time points
        x: x-coordinate
        y: y-coordinate
        v_x: x-component of (instantaneous) velocity
        v_y: y-component of (instantaneous) velocity
        v: (instantaneous) speed [omitted if incl_speed is False]
        in: whether object resides in ROI [omitted if marked_region is None]
      2/ a python list (if coords.ndim==2) or list of list (if coords.ndim==3)
        consisting of the above labels of the columns of the numpy array, with
        object index (beginning at 1) included if coords.ndim==3 (e.g., t1 
        for the time point column of the first object)
      3/ if summary is True, an additional named tuple (if coords.ndim==2) or
        list of named tuple (if coords.ndim==3). The attributes of the named
        tuple in increasing index are:
          .distance: total distance traveled [ = np.nan if incl_speed is False]
          .proportion: proportion time spent in region of interest 
            [ = np.nan if marked_region is None]
          
    NOTES: if smooth is False the velocities computation is routed to 
      calculate_motion(). If smooth is True the position and velocity 
      computation is routed to estimate_motion()
    '''
    
    if smooth:
        outarray = estimate_motion(
            coords, dt, window, pad_value, t0=t0, incl_speed=incl_speed
        )
        head = window // 2
        tail = coords.shape[0] - head
        v_tail = tail
        
    else:
        
        # time points
        t_array = np.multiply.outer(
            np.arange(len(coords)) * dt + t0, 
            np.ones((*coords.shape[1:-1], 1))
        )
        
        # velocity and speed, with last entry padded
        vels = compute_velocities(coords, dt, also_speed=incl_speed)
        vels = np.append(
            vels, np.full( (1, *vels.shape[1:]), pad_value), axis=0
        )
        
        outarray = np.concatenate( (t_array, coords, vels), axis=-1 )
        head = 0
        tail = coords.shape[0]
        v_tail = tail - 1
        
    # header output
    header = ["t{}", "x{}", "y{}", "v{}_x", "v{}_y"]
    if incl_speed: header.append("v{}")
    
    if marked_region is not None:
        
        # containment flag
        in_region = np.expand_dims(
            np.apply_along_axis(
                marked_region.contains_point, -1, outarray[..., 1:3]
            ), -1
        )
        
        # build output array
        outarray = np.concatenate((outarray, in_region), axis=-1)
        if smooth:
            outarray[:head, ..., -1] = pad_value
            outarray[tail:, ..., -1] = pad_value
        
        # append to header
        header.append("in{}")
    
    # convert header template to actual header
    if coords.ndim == 2:
        header = [ __.format("") for __ in header]
    else:
        n_obj = coords.shape[1]
        header = [ 
            [__.format(i) for __ in header] for i in range(1, n_obj + 1) 
        ]
    
    if summary:
        
        # find distance and proportion in ROI for all objects
        dist = np.sum(outarray[head:v_tail,...,5], axis=0) * dt if (
            incl_speed
        ) else np.sum(np.full(vels.shape[1:], np.nan), axis=-1)
        
        prop = np.sum(in_region[head:tail,...,0],axis=0)/(tail - head) if (
            marked_region is not None
            ) else np.sum(np.full(coords.shape[1:], np.nan), axis=-1)
        
        # pack into MotionSummary named tuples
        if coords.ndim == 2:
            summarized = MotionSummary(dist, prop)
        else:
            summarized = [ 
                MotionSummary(_1, _2) for _1, _2 in zip(dist, prop)
            ]
        
        return (outarray, header, summarized)
    
    else:
        return (outarray, header)

def detect_slowness(time, speed, speed_ubound, time_lbound, *, details=True):
    '''
    detect if an object is moving slowly for prolonged duration 
    (a.k.a. "freezes")
    
    ARGUMENTS:
      time: 1-dim numpy array of time points
      speed: 1-dim numpy array of the speed of the object at the respective 
        time points
      speed_ubound: the upper bound of speed below which the object is 
        considered slow
      time_lbound: the lower bound of time above which the object is 
        considered prolongedly slow
      details: whether to create detailed outputs
      
    RETURNS:
      if details is True: a 2-dim numpy array, with the 0th index being
      index of occurrences and 1st index follows the order of [start_time, 
      end_time, duration]
      if details is False: a 1-dim numpy array of the duration of prolonged
      slowness
      
    NOTES:
      expects the time and speed array to be of same length, but disregard
      the last data point in speed, in agreement with the output of
      assemble_motion_data()
    '''
    
    # idea: first create bool array indicating when speed is below upper bound
    # the diff of that array is an array of "trigger", where 1 indicates
    # onset of slowness and -1 indicates offset of slowness
    diff_array = np.zeros_like(time, dtype=int)
    diff_array[1:] = np.diff(np.array(speed < speed_ubound, dtype=int))
    if speed[0] < speed_ubound: diff_array[0] = 1 
    diff_array[-1] = -1 if (speed[-2] < speed_ubound) else 0
    
    start = time[np.nonzero(diff_array==1)]
    end = time[np.nonzero(diff_array==-1)]
    duration = end - start
    
    if details:
        out = np.transpose(np.array([start, end, duration]))
        return out[duration > time_lbound, :]
    else:
        return duration[duration > time_lbound]

#### I/O related codes

def imshow(im_array, *, mode=None, **kwargs):
    '''
    convert a numpy or imageio array into a PIL image object then display 
    the image object using PIL's show() method
    
    ARGUMENTS:
      im_array: a 2-dim (3-dim if colorspace) memory view of numpy array
      mode: PIL image mode to use (automatically determined if None)
      (**kwargs): remaining keyword arguments are passed to the function 
        PIL.Image.show()
    
    RETURNS none
    
    SIDE EFFECTS:
      image is display via the .show() method on PIL object
    '''
    img = PIL.Image.fromarray(im_array, mode=mode)
    img.show(**kwargs)

def make_cmap(spec, *, name="custom_cmap"):
    '''
    Create a custom matplotlib colormap ("cmap") based on the provided 
    specification
    
    ARGUMENTS:
      spec: the specification of matplotlib cmap
      name: the name of the resulting cmap
    
    RETURNS: 
      a matplotlib colormap object
    
    DETAILS:
      In decreasing priority, the input spec is interpreted as follows:
      1/ if spec is a cmap, it is returned and nothing else is done
      2/ if spec is a list or tuple:
        2.1/ If spec is of the form (original, start, stop), where original
          is a cmap or can be interpreted as cmap by get_cmap and start and 
          stop can be cast as float, then spec is treated as specifying a 
          "slice" of cmap from start to stop
        2.2/ If spec is a sequence of strings, each being color-like, then
          spec is treated as specifying a linear-segment cmap whose steps are
          given by the the strings
      3/ if spec is a string:
        3.1/ If the string can be interpreted as cmap by get_cmap, then spec 
          is treated as specifying that particular cmap
        3.2/ If the string is color-like, the cmap is treated as a uniform
          cmap of that color
      In all other cases ValueError is raised
    '''
    if isinstance(spec, mpl.colors.Colormap):
        return spec
    
    if isinstance(spec, list) or isinstance(spec, tuple):
        
        if all( isinstance(__, str) for __ in spec ):
            return mpl.colors.LinearSegmentedColormap.from_list(name, spec)
        
        else:
            if isinstance(spec[0], mpl.colors.Colormap):
                base_cmap = spec[0]
            else:
                base_cmap = plt.get_cmap(spec[0])
            
            start = float(spec[1])
            stop = float(spec[2])
            
            return mpl.colors.LinearSegmentedColormap.from_list(
                name, base_cmap(np.linspace(start, stop, 256))
            )
    
    if isinstance(spec, str):
        try:
            return  plt.get_cmap(spec)
        except ValueError:
            return mpl.colors.LinearSegmentedColormap.from_list(
                name, [spec, spec]
            )
    
    raise ValueError("invalid cmap specification")

def overlay_path(
    im_array, points, px2real=1.0, offset=(0, 0), closed=True, *, 
    cmap="gray", colorspace="RGB", facecolor=None, fc="none", 
    edgecolor=None, ec="black", linestyle=None, ls=":", 
    linewidth=None, lw=1, **kwargs
):
    '''
    create a new image array corresponding to the original image with the
    path represented by vertices added
    
    ARGUMENTS:
      im_array: a 2-dim (3-dim if colorspace) memory view of numpy array
      points: vertices (x_i, y_i) defining the boundary, as a sequence of 
        sequences (e.g., list of tuples); or a matplotlib path object
        Assumed to be expressed in physical coordinates and relative to the
        cropped frame (see px2real and offset below)
      px2real: conversion factor from pixel coordinates to physical coordinates
      offset: the offset (in pixel coordinate) between origin of the cropped
        frame for which the path is expressed in, and the origin of image, 
        in (left, top) format
      closed: whether the points represent a closed path. Ignored if
        points is a matplotlib path object
      cmap: color map used for converting im_array into matplotlib image; must
        be a valid specification to make_cmap()
      colorspace: the color space of output image. Possible choices are "L",
        "RGB", and "RGBA" (default to "RGB" if other values are supplied)
      facecolor OR fc: the facecolor of the region enclosed by path (if None
        the matplotlib default is used). If both are specified fc takes 
        precedence
      edgecolor OR ec: the edgecolor of the path (if None the matplotlib 
        default is used). If both are specified ec takes precedence
      linestyle OR ls: the linestyle of the path (if None the matplotlib 
        default is used). If both are specified ls takes precedence
      linewidth OR lw: the line-width of the added path, in units of pixel 
        (if None the matplotlib default is used). If both are specified lw
        takes precedence
      (**kwargs): remaining keyword arguments are passed to the function 
        matplotlib.patches.PathPatch() that constructs the path overlay
    
    RETURNS:
      a numpy array corresponding to the image with the added overlay
    
    NOTE:
      1/ the overlaid image is created via matplotlib backend. As a result,
        even in the null case (no overlay added) the output is likely to be
        slightly different from the input
      2/ offset can if fact contain more component. In particular, it can be
        a PIL-style cropbox
    '''
    
    # create actual cmap from specification
    cmap = make_cmap(cmap)
    
    # collapse the aliases for matplotlib options
    fc = facecolor if (fc is None) else fc
    ec = edgecolor if (ec is None) else ec
    ls = linestyle if (ls is None) else ls
    lw = linewidth if (lw is None) else lw
    
    # convert linewidth to unit of pixel
    if lw is not None:
        lw = lw * 72
    
    # attempt to convert path to mpl path if not already such
    if not isinstance(points, mpl.path.Path):
        path = vertices_to_path(points, scale = 1/px2real, closed=closed)
    else: # scale the path from physical to pixel coordinates
        transform = mpl.transforms.Affine2D().scale(1/px2real)
        path = points.transformed(transform)
    
    # translate the path by offset
    transform = mpl.transforms.Affine2D().translate(offset[0], offset[1])
    path = path.transformed(transform)
    
    # initialize a figure object outside of pyplot
    size = (im_array.shape[1], im_array.shape[0])
    fig = mpl.figure.Figure(figsize=size, dpi=1)
    canvas = mpl.backends.backend_agg.FigureCanvas(fig)
    
    # create Axes (plot area) the same size as the figure
    ax = fig.subplots()
    ax.set_position([0, 0, 1, 1], which='both')
    
    # place the image and the overlay
    ax.imshow(im_array, cmap=cmap)
    ax.set_axis_off()
    patch = mpl.patches.PathPatch(path, fc=fc, ec=ec, ls=ls, lw=lw, **kwargs)
    ax.add_patch(patch)
    
    # convert to (RGBA uint8) numpy array 
    canvas.draw()
    out = np.array(canvas.renderer.buffer_rgba())
    
    # color mode conversion and output
    colorspace = colorspace.upper()
    if colorspace=="RGBA":
        return out
    else:
        out = skm.color.rgba2rgb(out)
        if colorspace=="L":
            return (skm.color.rgb2gray(out) * 255).astype(np.uint8)
        else:
            return (out * 255).astype(np.uint8)

def overlay_coords(
    im_array, coords, px2real=1.0, offset=(0, 0), 
    color="green", radius=5, opacity=0.5
):
    '''
    create a new image array corresponding to the original image with a 
    particular coordinate highlighted
    
    ARGUMENTS:
      im_array: a 2-dim (3-dim if colorspace) memory view of numpy array
      coords: a 2-component numpy array of the coordinates at which the 
        highlighting disk centers at. Assumed to be expressed in physical 
        coordinates and relative to the cropped frame (see px2real and 
        offset below)
      px2real: conversion factor from pixel coordinates to physical coordinates
      offset: the offset (in pixel coordinate) between origin of the cropped
        frame for which the path is expressed in, and the origin of image, 
        in (left, top) format
      color: the color of the highlighting disk. Either a named color in 
        PIL.ImageColor or a sequence of 3 (uint8) integers as RGB values
      radius: the radius (in pixel) of the highlighting disk. If radius < 1
        it is interpreted as measure in units of the SHORTER side of the image
      opacity: the opacity of the overlaid highlighting disk, expressed as 
        a number between 0 and 1
    
    RETURNS:
      a numpy array corresponding to the image with the added overlay
      
    NOTES: 
      offset can if fact contain more component. In particular, it can be
      a PIL-style cropbox
    '''
    # interpret radius is it is a float < 1
    if radius < 1:
        radius = max(1, int(radius * min(im_array.shape)))
    
    # interpret color
    if type(color)==str:
        color = np.array(PIL.ImageColor.getrgb(color))
    else:
        np.array(color)
    
    # convert im_array to RGB color-space
    if im_array.dtype==bool:
        im_array = 255 * im_array.astype(np.uint8)
    if im_array.ndim < 3:
        im_array = skm.color.gray2rgb(im_array)
    
    # find the pixels that need to be modified
    # NOTE: the ::-1 is needed to convert (x,y) to (row, col) coords
    center = np.rint(coords / px2real + offset[:2]).astype(int)
    rr, cc = skm.draw.disk(center[::-1], radius, shape=im_array.shape[:2])
    
    # modified the said pixels
    im_array[rr, cc] = (
        im_array[rr, cc] * (1 - opacity) + opacity * color
    ).astype(np.uint8)
    
    return im_array

def export_overlaid_video(
    filename, frames, coords_array, px2real=1.0, offset=(0, 0), 
    color="green", radius=10, opacity=0.5, trail=False, trail_color="white", 
    trail_opacity=0.5, trail_mask = [(0,0), (1,0), (0,1), (-1,0), (0, -1)], 
    **kwargs
):
    '''
    export a video in which the original frames are augmented by highlighting
    disks centered at location specified by coords_array. In addition, a 
    trail of the motion can also be included
    
    ARGUMENTS:
      filename: filename of the output file (extension included)
      frames: an iterator of 2-dim (3 with colorspace?) numpy array 
        representing a sequence of images, with shape ~ (row, col, [color])
      coords_array: a sequence of 2-component numpy array (x_i, y_i) of 
        coordinates at which the highlighting disk centers at. Assumed to be 
        expressed in physical coordinates and relative to the cropped frame 
        (see px2real and offset below)
      px2real: conversion factor from pixel coordinates to physical coordinates
      offset: the offset (in pixel coordinate) between origin of the cropped
        frame for which the path is expressed in, and the origin of image, 
        in (left, top) format 
      color: the color of the highlighting disk. Either a named color in 
        PIL.ImageColor or a sequence of 3 (uint8) integers as RGB values
      radius: the radius (in pixel) of the highlighting disk
      opacity: the opacity of the overlaid highlighting disk, expressed as 
        a number between 0 and 1
      trail: whether to also overlay the trail of motion
      trail_color: color for the overlaid trail. Same convention as color
      trail_opacity: opacity of the overlaid trail. Same convention as opacity
      trail_mask: a sequence of 2-tuple of integers. The line segments forming
        the trail is duplicate at each offset defined by each 2-tuple. This is
        a poor person's way to create "bold" line segments
      (**kwargs): all remaining keyword arguments are passed to the 
        imageio writer (e.g., frame rate via fps=...)
    
    RETURNS None
    
    SIDE EFFECTS: overlaid frames written as a file named filename
      
    NOTES:
      offset can if fact contain more component. In particular, it can be
      a PIL-style cropbox
    '''
    # make offset efficient
    offset = np.array(offset[:2])
    
    # resolve named colors
    if type(color)==str:
        color = np.array(PIL.ImageColor.getrgb(color))
    else:
        color = np.array(color)
    
    # initialize trail related information
    if trail:
        if trail_mask is None:
            trail_mask = [ (0, 0) ]
        if type(trail_color)==str:
            trail_color = np.array(PIL.ImageColor.getrgb(trail_color))
        else:
            trail_color = np.array(trail_color)
        trail_rr = np.zeros( (0,), dtype=int)
        trail_cc = np.zeros( (0,), dtype=int)
        
    writer = imageio.get_writer(filename, **kwargs)
    
    old_pixel = None
    for frame, coords in zip(frames, coords_array):
        
        # convert frame to RGB color
        if frame.dtype==bool:
            frame = 255 * frame.astype(np.uint8)
        if frame.ndim < 3:
            frame = skm.color.gray2rgb(frame)
        
        # cache the shape of the frame
        shape = frame.shape[:2]
        
        # sidestep any nan's
        if np.any(np.isnan(coords[:2])):
            
            # overlay existing trail to image
            if trail:
                old_pixel = None
                frame[trail_rr, trail_cc] = (
                    frame[trail_rr, trail_cc] * (1 - trail_opacity) + 
                    trail_opacity * trail_color
                ).astype(np.uint8)
            
            writer.append_data(frame)
            continue
        
        # pixel coordinate of object location
        # NOTE: the ::-1 is needed to convert (x,y) to (row, col) coords
        pixel = np.rint(coords / px2real + offset).astype(int)[::-1]
        
        if trail:
        
            # update trail if there is a previous known position
            if old_pixel is not None:
                rr, cc = skm.draw.line(*old_pixel, *pixel)
                for (x, y) in trail_mask:
                    trail_rr = np.append(trail_rr, rr + y)
                    trail_cc = np.append(trail_cc, cc + x)
            
            # overlay trail to image
            frame[trail_rr, trail_cc] = (
                frame[trail_rr, trail_cc] * (1 - trail_opacity) + 
                trail_opacity * trail_color
            ).astype(np.uint8)
        
        # overlay current position to image
        rr, cc = skm.draw.disk(pixel, radius, shape=shape)
        frame[rr, cc] = (
            frame[rr, cc] * (1 - opacity) + opacity * color
        ).astype(np.uint8)
        
        writer.append_data(frame)
        old_pixel = pixel # store last pixel coordinates
    
    writer.close()

def export_overlaid_n_video(
    filename, frames, coords_array, px2real=1.0, offset=(0, 0), 
    colors="green", radii=10, opacity=0.5, trails=False, trail_colors="black", 
    trail_opacity=0.5, trail_mask = [(0,0), (1,0), (0,1), (-1,0), (0, -1)], 
    **kwargs
):
    '''
    export a video in which the original frames are augmented by highlighting
    disks centered at location specified by coords_array. In addition, trails
    of the motion can also be included
    
    ARGUMENTS:
      filename: filename of the output file (extension included)
      frames: an iterator of 2-dim (3 with colorspace?) numpy array 
        representing a sequence of images, with shape ~ (row, col, [color])
      coords_array: a sequence (~ frame number) of sequence (~ labels of
        objects) of 2-component numpy array ( x_i(t), y_i(t) ) of coordinates 
        at which the i-th highlighting disk centers at when the frame is at 
        time t. The values of ( x_i(t), y_i(t) ) are expressed in physical 
        coordinates and relative to the cropped frame (see px2real and 
        offset below)
      px2real: conversion factor from pixel coordinates to physical coordinates
      offset: the offset (in pixel coordinate) between origin of the cropped
        frame for which the path is expressed in, and the origin of image, 
        in (left, top) format 
      colors: the colors of the highlighting disk. A sequence of either
        named colors in PIL.ImageColor or 1-dim numpy arrays of 3 (uint8) 
        integers as RGB values. The index of the sequence corresponds to the 
        label of objects. The sequence need not has the same length as the 
        number of objects and will be cycled through if needed. As a fallback 
        a single color in PIL.ImageColor or a 1-dim numpy array can be given 
        and will be interpreted as specification for all objects
      radii: a sequence giving the radii (in pixel) of the highlighting disk,
        with index running through object labels. The sequence need not has the
        same length as the number of objects and will be cycled through if 
        needed. If scalar all highlighting disks are taken to have the 
        same radius
      opacity: the opacity of the overlaid highlighting disk, expressed as 
        a number between 0 and 1
      trails: whether to also overlay the trails of motion
      trails_color: colors for the overlaid trail. Same convention as colors
      trail_opacity: opacity of the overlaid trail. Same convention as opacity
      trail_mask: a sequence of 2-tuple of integers. The line segments forming
        the trail is duplicate at each offset defined by each 2-tuple. This is
        a poor person's way to create "bold" line segments
      (**kwargs): all remaining keyword arguments are passed to the 
        imageio writer (e.g., frame rate via fps=...)
    
    RETURNS None
    
    SIDE EFFECTS: overlaid frames written as a file named filename
      
    NOTES:
      offset can if fact contain more component. In particular, it can be
      a PIL-style cropbox
    '''
    # identify number of objects
    n_obj = coords_array.shape[1]
    
    # make offset efficient
    offset = np.array(offset[:2])
    
    # resolve colors (str -> np.array and scalar -> list)
    if type(colors)==str:
        colors = [ np.array(PIL.ImageColor.getrgb(colors)) ]
    elif isinstance(colors, np.ndarray) and colors.ndim==1:
        colors = [ colors ]
    else:
        colors = [ np.array( 
            (PIL.ImageColor.getrgb(__) if (type(__)==str) else __) 
        ) for __ in colors ]
        
    # resolve radii into list
    try:
        iter(radii)
    except TypeError:
        radii = [ radii ]
    
    # initialize trail related information
    if trails:
        if trail_mask is None:
            trail_mask = [ (0, 0) ]
        if type(trail_colors)==str:
            trail_colors = [ np.array(PIL.ImageColor.getrgb(trail_colors)) ]
        elif isinstance(trail_colors, np.ndarray) and trail_colors.ndim==1:
            trail_colors = [ trail_colors ]
        else:
            trail_colors = [ np.array( 
                (PIL.ImageColor.getrgb(__) if (type(__)==str) else __) 
            ) for __ in trail_colors ]
        trails_rr = [ np.zeros( (0,), dtype=int) ] * n_obj
        trails_cc = [ np.zeros( (0,), dtype=int) ] * n_obj
    else:
        trail_colors = [ None ] # dummy value
    
    writer = imageio.get_writer(filename, **kwargs)
    
    old_pixels = [ None ] * n_obj
    for frame, coords_n in zip(frames, coords_array):
        
        # convert frame to RGB color
        if frame.dtype==bool:
            frame = 255 * frame.astype(np.uint8)
        if frame.ndim < 3:
            frame = skm.color.gray2rgb(frame)
        
        # cache the shape of the frame
        shape = frame.shape[:2]
        
        for i, (coords, color, radius, trail_color, old_pixel) in enumerate(
            zip(
                coords_n, itertools.cycle(colors), itertools.cycle(radii), 
                itertools.cycle(trail_colors), old_pixels
        )):
            
            # sidestep any nan's
            if np.any(np.isnan(coords[:2])):
                
                # overlay existing trails to image
                old_pixels[i] = None
                if trails:
                    trail_rr = trails_rr[i]
                    trail_cc = trails_cc[i]
                    frame[trail_rr, trail_cc] = (
                        frame[trail_rr, trail_cc] * (1 - trail_opacity) + 
                        trail_opacity * trail_color
                    ).astype(np.uint8)
                
                continue
            
            # pixel coordinate of object location
            pixel = np.rint(coords / px2real + offset).astype(int)[::-1]

            if trails:
                
                # update trails if there is a previous known position
                if old_pixel is not None:
                    rr, cc = skm.draw.line(*old_pixel, *pixel)
                    for (x, y) in trail_mask:
                        trail_rr = np.append(trails_rr[i], rr + y)
                        trail_cc = np.append(trails_cc[i], cc + x)
                    trails_rr[i] = trail_rr
                    trails_cc[i] = trail_cc
                else:
                    trail_rr = trails_rr[i]
                    trail_cc = trails_cc[i]
                
                # overlay trail to image
                frame[trail_rr, trail_cc] = (
                    frame[trail_rr, trail_cc] * (1 - trail_opacity) + 
                    trail_opacity * trail_color
                ).astype(np.uint8)
                
            # update old_pixels 
            # this line must be OUTSIDE the old_pixel conditional...
            old_pixels[i] = pixel
            
            # overlay current position to image
            rr, cc = skm.draw.disk(pixel, radius, shape=shape)
            frame[rr, cc] = (
                frame[rr, cc] * (1 - opacity) + opacity * color
            ).astype(np.uint8)
        
        writer.append_data(frame)
    
    writer.close()

def export_video(filename, frames, **kwargs):
    '''
    export a given sequence of frames as an mp4 video via imageio
    
    ARGUMENTS:
      filename: filename of the output file (extension included)
      frames: 3-dim (4 with colorspace?) numpy array representing a
        sequence of images, with the 0th index being the sequence index
      (**kwargs): all remaining keyword arguments are passed to the 
        imageio writer (e.g., frame rate via fps=...)
      
    RETURNS None
    
    SIDE EFFECT: data from frames written to file named filename
    '''
    
    writer = imageio.get_writer(filename, **kwargs)
    
    # extract dtype of frames (peek into first frame if needed)
    try:
        dtype = frames.dtype
    except AttributeError:
        frame0, frames = _peek_iterator(frames)
        dtype = frame0.dtype
    
    if dtype==np.uint8: # no conversion needed
        for im in frames:
            writer.append_data(im)
    elif dtype==bool: # binary data: boost contrast to max
        for im in frames:
            writer.append_data(255*im.astype("uint8"))
    else: # fallback: blindly convert to uint8
        for im in frames:
            writer.append_data(im.astype("uint8"))
    
    writer.close()

def export_image(filename, frame, mode=None, **kwargs):
    '''
    export a given frame as an image file via PIL
    
    ARGUMENTS:
      filename: filename of the output file (extension included)
      frame: 2-dim (3 with colorspace) numpy array representing a
        single image
      mode: PIL image mode to use (automatically determined if None)
      (**kwargs): all remaining keyword arguments are passed to the 
        PIL's .save() function
      
    RETURNS None
    
    SIDE EFFECT: data from frame written to file named filename
    '''
    dtype = frame.dtype
    if dtype==np.uint8: # no conversion needed
        pass
    elif dtype==bool: # binary data: boost contrast to max
        frame = 255 * frame.astype("uint8")
    else: # fallback: blindly convert to uint8
        frame = frame.astype("uint8")
    
    img = PIL.Image.fromarray(frame, mode=mode)
    img.save(filename, **kwargs)

def write_csv(filename, data, header=None):
    '''
    write given data into csv
    
    ARGUMENTS:
      filename: filename of the output file (extension included)
      frames: 2-dim numpy array of data, with 0th (1st) index corresponds
        to rows (columns) in csv
      header: the header row for the csv as a sequence (e.g. list); if None
        no header row is written
      
    RETURNS None
    
    SIDE EFFECTS: data written to file named filename
    '''
    
    with open(filename, "w", newline='') as outfile:
        csv_writer = csv.writer(outfile)
        if header is not None:
            csv_writer.writerow(header)
        for row in data:
            csv_writer.writerow(row)

def read_csv(filename, header=True, skip=0, parser=None, processor=None):
    '''
    read csv and return the result as a list of list (or transformed and/or 
    post-processed outcome thereof)
    
    ARGUMENTS:
      filename: the filename of the csv to be read
      header: whether there is a header row to be read differently and returned
      skip: the number of row to skip at the beginning, AFTER the header row
        if applicable
      parser: None or callable
        if None, the rows are read as is (therefore each entry is a string)
        if function, the function is applied to each element in the row before
        they are assembled
      processor: None or callable
        if None, the parsed and assembled row are assemble into an outer list
        if callable, each row is passed to the processor and the result
        is assembled into an outer list
        
      RETURNS:
        if processor is None, a list of list representing the bulk of the 
        content in the csv file; if processor is not None the inner list 
        is replaced by its processed counterpart;
        if header is True, an additional python list consisting of the entry
        in the header row
    '''
    
    out_table = []
    with open(filename, "r") as infile:
        
        # use manual next() to advance rows
        csv_reader = csv.reader(infile)
        if header: 
            header_row = next(csv_reader)
        for row in range(skip): 
            next(csv_reader)
        
        # keep the for loop out of conditional for performance's sake
        if parser is None:
            if processor is None:
                for row in csv_reader:
                    out_table.append(row)
            else:
                for row in csv_reader:
                    out_table.append(processor(row))
        else:
            if processor is None:
                for row in csv_reader:
                    out_table.append([parser(__) for __ in row])
            else:
                for row in csv_reader:
                    out_table.append(processor([parser(__) for __ in row]))
    
    if header:
        return (out_table, header_row)
    else:
        return out_table

def plot_path(
    title, coords, t_range=None, region=None, info_list=None,
    height_ratios=(8,1,1), save_as=None, close=False, *,
    cmap="cool", x_lim=None, y_lim=None, 
    x_label="x-position", y_label="y-position", t_label="time", 
    figsize=(8, 10), fontsizes=(16, 12, 12)
):
    '''
    create a plot of path taken by the object being tracked
    
    ARGUMENTS:
      title: the title of the plot
      coords: 2-dim numpy array representing a sequence of coordinates, 
        with the 0th index being the sequence index
      t_range: a 2-tuple consisting of the time point at the beginning and
        the end of the motion, respectively, or None if not known
      region: a matplotlib path representing the boundary of a region
        of interest
      info_list: list of strings of additional info to be included
        (each string will occupy one line in the figure)
      height_ratio: ratio of heights between main plot, color map, and plot 
        region for additional info [last item omitted if info_list is None]
      save_as: filename (extension included) to save the plot to 
        (if None the plot is not saved)
      close: whether to immediately close the plot (i.e., no visual output)
      cmap: color map used to indicate time-point that corresponds to a 
        particular spatial coordinates. Has to be a valid specification to 
        make_cmap()
      x_lim: the left and right x-axis view limits as a sequence, or None
        if the x-axis view is to be set automatically
      y_lim: the top and bottom y-axis view limits as a sequence, or None
        if the y-axis view is to be set automatically
      x_label: label for x-axis in the main plot
      y_label: label for y-axis in the main plot
      t_label: label for time- (color-) axis
      figsize: overall size of figure (in inches?)
      fontsizes: fontsizes used in (0th index) title, (1st index) legends, 
        and (2nd index) ticks
    
    RETURNS:
      matplotlib figure object representing the resulting plot
    
    SIDE EFFECTS:
      if save_as is not None an image file is saved
      depending on matplotlib backend the plot may be displayed
    '''
    
    # convert cmap specification to actual cmap
    cmap = make_cmap(cmap)
    
    # instantize height_ratios and set last index appropriately
    if info_list is None:
        # plot as vertical stack of 2 subfigures (axes objects)
        height_ratios = list(height_ratios[:-1])
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, figsize=figsize, constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios}
        )
    else:
        # plot as vertical stack of 3 subfigs
        height_ratios = list(height_ratios)
        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=3, figsize=figsize, constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios}
        )
    
    # first subfig is the plot of path in 2D space
    if x_lim is not None:
        ax1.set_xlim(*x_lim)
    if y_lim is not None:
        ax1.set_ylim(*y_lim)
    ax1.set_aspect("equal") # 1-to-1 scale for paths
    ax1.invert_yaxis()
    ax1.set_title(title, fontsize=fontsizes[0])
    ax1.tick_params(axis='both', which='major', labelsize=fontsizes[2])
    ax1.set_xlabel(x_label, fontsize=fontsizes[1])
    ax1.set_ylabel(y_label, fontsize=fontsizes[1])
    
    # include region of interest as dotted lines
    if region is not None:
        patch = mpl.patches.PathPatch(region, fc='none', ls=":", lw=2)
        ax1.add_patch(patch)
    
    # plot the path in solid lines
    ax1.plot(coords[:,0], coords[:,1], "k-")
    
    # plot coordinates at time points as points of varying colors
    c_array = cmap(np.linspace(0, 1, len(coords)))
    ax1.scatter(coords[:,0], coords[:,1], c=c_array)

    # second subfig is a strip to illustrate color map of time points
    c_strip = np.tile(np.linspace(0, 1, 256), 2).reshape((2, -1))
    if t_range is None:
        extent = [0, 1, 0, 1]
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["start", "end"])
        ax2.tick_params(axis='x', which='major', labelsize=fontsizes[1])
    else:
        extent = [t_range[0], t_range[1], 0, 1]
        ax2.tick_params(axis='x', which='major', labelsize=fontsizes[2])
    ax2.imshow(c_strip, aspect='auto', extent=extent, cmap=cmap)
    ax2.set_ylim([-0, 1.2])
    for direction in ["left", "right", "top", "bottom"]:
        ax2.spines[direction].set_visible(False)
    ax2.axes.xaxis.set_visible(True)
    ax2.axes.yaxis.set_visible(False)
    ax2.set_xlabel(t_label, fontsize=fontsizes[1])
    
    # third subfig prints the information from info_list
    if info_list is not None:
        ax3.set_axis_off()
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, len(info_list)])
        ax3.invert_yaxis()
        for (i, entry) in enumerate(info_list):
            ax3.text(0, i, entry, ha="left", fontsize=fontsizes[1])
    
    # save figure if asked
    if save_as is not None:
        fig.savefig(save_as, bbox_inches = 'tight')
    
    # close if asked
    if close:
        plt.close()

    return fig

def plot_n_paths(
    title, coords, t_range=None, region=None, info_list=None,
    height_ratios=(8,1,1), save_as=None, close=False, *,
    cmaps="cool", linecolors="black", markers = "o", obj_labels="object #{}", 
    x_lim=None, y_lim=None, x_label="x-position", y_label="y-position", 
    t_label="time", figsize=(8, 10), fontsizes=(16, 12, 12)
):
    '''
    create a plot of the paths (one for each object) taken by the objects 
    being tracked
    
    ARGUMENTS:
      title: the title of the plot
      coords: 3-dim numpy array representing a sequence of coordinates, 
        with the 0th index the sequence index and the 1st index the 
        object index
      t_range: a 2-tuple consisting of the time point at the beginning and
        the end of the motion, respectively, or None if not known
      region: a matplotlib path representing the boundary of a region
        of interest
      info_list: list of strings of additional info to be included
        (each string will occupy one line in the figure)
      height_ratio: ratio of heights between main plot, color map, and plot 
        region for additional info [last item omitted if info_list is None]
      save_as: filename (extension included) to save the plot to 
        (if None the plot is not saved)
      close: whether to immediately close the plot (i.e., no visual output)
      cmaps: sequence of color maps that indicate the time point that
        corresponds to a particular spatial coordinates for the i-th object.
        Each element in the sequence has to be valid specification to 
        make_cmap(). The sequence need not has the same length as the number 
        of objects and will be cycled through if needed. As a fallback a single
        cmap can be given and will be interpreted as specification for all
        objects
      linecolors: sequence of matplotlib color specification for the linecolor
        for the i-th object. The sequence need not has the same length as the 
        number of objects and will be cycled through if needed. As a fallback
        a single color can be specified and will be interpreted as 
        specifications for all objects
      markers: sequence of matplotlib marker specification for the i-th object.
        The sequence need not has the same length as the number of objects and
        will be cycled through if needed. As a fallback a single color can be 
        specified and will be interpreted as specifications for all objects
      obj_labels: sequence of labels for the ith object being tracked. If a 
        single string is provided it will be passed to .format(i), with i
        being integers starting from 1, to general the label for each object
      x_lim: the left and right x-axis view limits as a sequence, or None
        if the x-axis view is to be set automatically
      y_lim: the top and bottom y-axis view limits as a sequence, or None
        if the y-axis view is to be set automatically
      x_label: label for x-axis in the main plot
      y_label: label for y-axis in the main plot
      t_label: label for time- (color-) axis
      figsize: overall size of figure (in inches?)
      fontsizes: fontsizes used in (0th index) title, (1st index) legends, 
        and (2nd index) ticks
    
    RETURNS:
      matplotlib figure object representing the resulting plot
    
    SIDE EFFECTS:
      if save_as is not None an image file is saved
      depending on matplotlib backend the plot may be displayed
    '''
    n_obj = coords.shape[1]
    
    # duplicate colormaps if only one is given
    # try to convert color-like to cmap if needed
    if isinstance(cmaps, list) or isinstance(cmaps, tuple):
        try:
            cmaps = [ 
                make_cmap(__, name="custom_cmap_{}".format(i)) for
                (i, __) in enumerate(cmaps)
            ]
            multi_cmap = True
        except Exception as e:
            cmaps = [ make_cmap(cmaps) ]
            multi_cmap = False
    else:
        cmaps = [ make_cmap(cmaps) ]
        multi_cmap = False
    
    # duplicate line color if only one is given
    if mpl.colors.is_color_like(linecolors):
        linecolors = [ linecolors ]
        multi_lc = False
    else:
        if len(linecolors) == 1:
            multi_lc = False
        else:
            multi_lc = True
        
    # duplicate markers if only one is given
    if isinstance(markers, list) or isinstance(markers, tuple):
        if len(markers) == 1:
            multi_markers = False
        else:
            multi_markers = True
    else:
        markers = [ markers ]
        multi_markers = False
    
    # create all labels if only a template string is given
    if isinstance(obj_labels, str):
        obj_labels = [ obj_labels.format(i) for i in range(1, n_obj+1) ]
    
    # instantize height_ratios and set last index appropriately
    if info_list is None:
        # plot as vertical stack of 2 subfigures (axes objects)
        height_ratios = list(height_ratios[:-1])
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, figsize=figsize, constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios}
        )
    else:
        # plot as vertical stack of 3 subfigs
        height_ratios = list(height_ratios)
        fig, (ax1, ax2, ax3) = plt.subplots(
            nrows=3, figsize=figsize, constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios}
        )
    
    # title of the entire plot
    fig.suptitle(title, fontsize=fontsizes[0])
    
    # first subfig is the plot of paths in 2D space
    if x_lim is not None:
        ax1.set_xlim(*x_lim)
    if y_lim is not None:
        ax1.set_ylim(*y_lim)
    ax1.set_aspect("equal") # 1-to-1 scale for paths
    ax1.invert_yaxis()
    ax1.tick_params(axis='both', which='major', labelsize=fontsizes[2])
    ax1.set_xlabel(x_label, fontsize=fontsizes[1])
    ax1.set_ylabel(y_label, fontsize=fontsizes[1])
    
    # include region of interest as dotted lines
    if region is not None:
        patch = mpl.patches.PathPatch(region, fc='none', ls=":", lw=2)
        ax1.add_patch(patch)
    
    # for each object, plot the path in solid lines
    for i, cmap, color, mark, label in zip(
        range(n_obj), itertools.cycle(cmaps), itertools.cycle(linecolors),
        itertools.cycle(markers), itertools.cycle(obj_labels)
    ):
        c_array = cmap(np.linspace(0, 1, len(coords)))
        ax1.plot(coords[:,i,0], coords[:,i,1], color=color, label=label)
        ax1.scatter(coords[:,i,0], coords[:,i,1], c=c_array, marker=mark)
    
    # plot the legend if the linecolor or marker is distinct for each object
    if multi_lc or multi_markers:
        
        legend_handles = []
        for i, cmap, color, mark, label in zip(
            range(n_obj), itertools.cycle(cmaps), itertools.cycle(linecolors),
            itertools.cycle(markers), itertools.cycle(obj_labels)
        ):
            legend_handles.append(mpl.lines.Line2D(
                    [0], [0], color=color, marker=mark,
                    mfc=cmap(0.5), mec="none", label=label
            ))
        
        ax1.legend(
            handles=legend_handles,
            loc='upper center', bbox_to_anchor=(0.5, 1.15), 
            ncol=n_obj, fontsize=fontsizes[2]
        )
    
    # second subfig is a strip to illustrate color map of time points
    c_strip = np.tile(np.linspace(0, 1, 256), 2).reshape((2, -1))
    
    # set x-axis depending on whether t_range is specified
    ax2.axes.xaxis.set_visible(True)
    ax2.set_xlabel(t_label, fontsize=fontsizes[1])
    if t_range is None:
        extent_base = [0, 1, 0, 1]
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["start", "end"])
        ax2.tick_params(axis='x', which='major', labelsize=fontsizes[1])
    else:
        extent_base = [t_range[0], t_range[1], 0, 1]
        ax2.tick_params(axis='x', which='major', labelsize=fontsizes[2])
    
    # set y-axis depending on whether there are distinct colormaps
    ax2.set_ylim([0, 1])
    ax2.invert_yaxis()
    if multi_cmap: # distinct colormap for each object
    
        # draw one strip for each object
        for i, cmap in zip(range(n_obj), itertools.cycle(cmaps)):
            extent = [ __ for __ in extent_base ]
            extent[2] = i / n_obj + 1 / (4 * n_obj)
            extent[3] = (i + 1)/ n_obj
            ax2.imshow(c_strip, aspect='auto', extent=extent, cmap=cmap)
            
        # use y-axis to display labels of objects
        ax2.axes.yaxis.set_visible(True)
        ax2.set_yticks([(8*i+5)/(8*n_obj) for i in range(n_obj)])
        ax2.set_yticklabels(obj_labels)
        ax2.tick_params(axis='y', which='major', labelsize=fontsizes[2])
    
    else: # same colormap for all objects => hide y-axis
        ax2.imshow(c_strip, aspect='auto', extent=extent_base, cmap=cmap)
        ax2.axes.yaxis.set_visible(False)
    
    # remove "spines" (frame) from the second plot
    for direction in ["left", "right", "top", "bottom"]:
        ax2.spines[direction].set_visible(False)
    
    # third subfig prints the information from info_list
    if info_list is not None:
        ax3.set_axis_off()
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, len(info_list)])
        ax3.invert_yaxis()
        for (i, entry) in enumerate(info_list):
            ax3.text(0, i, entry, ha="left", fontsize=fontsizes[1])
    
    # save figure if asked
    if save_as is not None:
        fig.savefig(save_as, bbox_inches = 'tight')
    
    # close if asked
    if close:
        plt.close()

    return fig

# control the import * behavior
__all__ = [
    # Image processing related codes
    'build_grayscaled', 'build_postprocessed', 'build_subtracted', 
    'build_thresholded', 'iter_grayscaled', 'iter_postprocess',
    'iter_postprocessed', 'iter_subtracted', 'iter_thresholded', 
    'calc_background', 'calc_threshold', 'count_intensity', 'morph',
    # Image region/path identification related codes
    'calc_pixel_to_phys', 'convert_to_physical', 'convert_to_pixel',
    'cropbox_to_slices', 'cropbox_to_vertices', 'mark_image', 
    'vertices_to_cropbox', 'vertices_to_mask', 'vertices_to_path',
    # Analysis related codes
    'assemble_motion_data', 'compute_centroids', 'compute_n_centroids', 
    'calculate_motion', 'detect_slowness', 'estimate_motion', 
    # I/O related codes
    'build_from_iter',
    'export_image', 'export_video', 'export_overlaid_video', 
    'export_overlaid_n_video', 'imshow', 'make_cmap', 'overlay_coords', 
    'overlay_path', 'plot_path', 'plot_n_paths', 'read_csv', 'write_csv'
    # experimental
]
