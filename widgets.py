'''
module to support interactive widgets in Jupyter environment

This module defines two functions, animate_frames() and animate_video(),
that can be called from within the Jupyter environment to create in-place 
widgets
'''

import ipywidgets as widgets
import matplotlib.pyplot as plt
import imageio

def animate_frames(
    frames, start=0, stop=None, step=1, *, subsampled=False, cmap="gray", 
):
    '''
    Interactively show particular frame from a sequence of frames in the 
    Jupyter environment, with a slider to adjust the frame being displayed
    
    ARGUMENTS:
      frames: a numpy array of sequence of numpy array representing the frames
        to be displayed
      start: the lower-bound (inclusive) of frame to be displayed
      stop: the upper-bound (exclusive) of frame to be displayed (None is
        equivalent to setting stop to be the number of frames)
      step: the increment in frame number
      subsampled: if True, the supplied frames is assumed to be a 
        start:stop:step slice from some underlying frames. All sub-sampled 
        frames will be reachable from the slider, but the slider index will
        align with the UNDERLYING frames rather than the sub-sampled frames
      cmap: the colormap used display the image (via matplotlib); can be 
        a matplotlib cmap object or a string for specifying one. Ignored
        if frames are in RGB(A)
        
    RETURNS:
      matplotlib figure object representing the resulting plot of frame
      
    NOTES:
      In Jupyter notebook, precede with magic command "%matplotlib notebook"
      In Jupyter lab, precede with magic command "%matplotlib ipympl"
    '''
    # resolve stop value
    if stop is None:
        if subsampled:
            stop = start + (len(frames) - 1) * step
        else:
            stop = len(frames) - 1
    
    # static information about the figure not modified by frame index
    fig, ax = plt.subplots()
    ax.set_position([0, 0, 1, 1], which='both')
    ax.set_axis_off()

    # initial view
    idx0 = 0 if subsampled else start
    img_obj = ax.imshow(frames[idx0], cmap=cmap)
    
    # inner function to display the figure
    # x and y are actually redundant. This is a trick to link the Play
    # widget to the IntSlider widget
    def f(x, y):
        if subsampled:
            y = (y - start) // step
        img_obj.set_data(frames[y])
    
    # interactively drive the inner function
    play = widgets.Play(description="n", min=start, max=stop, step=step, value=0)
    slider = widgets.IntSlider(description="n", min=start, max=stop, step=step, value=0)
    widgets.jslink((play, 'value'), (slider, 'value'))
    widgets.interact(f, x=play, y=slider)
    
    # return figure object so that it can be saved, closed, etc
    return fig

def animate_video(reader, start=0, stop=None, step=1, cmap="gray", **kwargs):
    '''
    Interactively show particular frame from a video in the Jupyter 
    environment, with a slider to adjust the frame being displayed
    
    ARGUMENTS:
      reader: an imageio reader object pointing to the video, or a path-like
        object (e.g., string) pointing to the location of the video file
      start: the lower-bound (inclusive) of frame to be displayed
      stop: the upper-bound (exclusive) of frame to be displayed (None is
        equivalent to setting stop to be the number of frames)
      step: the increment in frame number
      cmap: the colormap used display the image (via matplotlib); can be 
        a matplotlib cmap object or a string for specifying one. Ignored
        if video frames are in RGB(A)
        
    RETURNS:
      1/ matplotlib figure object representing the resulting plot of frame
      2/ the imageio reader object of the video (so that it can be closed
        properly after use)
      
    NOTES:
      In Jupyter notebook, precede with magic command "%matplotlib notebook"
      In Jupyter lab, precede with magic command "%matplotlib ipympl"
    '''
    
    # try to create imageio reader if user does not supply the reader object
    if not isinstance(reader, imageio.core.format.Format.Reader):
        reader = imageio.get_reader(reader)
    
    # resolve stop value
    if stop is None:
        stop = reader.count_frames() - 1
    
    # static information about the figure not modified by frame index
    fig, ax = plt.subplots()
    ax.set_position([0, 0, 1, 1], which='both')
    ax.set_axis_off()
    img_obj = ax.imshow(reader.get_data(0, **kwargs), cmap=cmap)
    
    # inner function to display the figure
    # x and y are actually redundant. This is a trick to link the Play
    # widget to the IntSlider widget
    def f(x, y):
        img_obj.set_data(reader.get_data(y, **kwargs))
    
    # interactively drive the inner function
    play = widgets.Play(description="n", min=start, max=stop, step=step, value=0)
    slider = widgets.IntSlider(description="n", min=start, max=stop, step=step, value=0)
    widgets.jslink((play, 'value'), (slider, 'value'))
    widgets.interact(f, x=play, y=slider)
    
    # return figure/reader object so that they can be saved, closed, etc
    return fig, reader

__all__ = [animate_frames, animate_video]