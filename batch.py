'''
utilities for batch processing videos for motion analysis

The main purpose of this module is to define two classes for batch processing,
namely SingleObjBatch (for single foreground object tracking) and MultiObjBatch
(for multi foreground object tracking). For both classes, the main usage 
pattern involve initializing an instance with particular configuration, 
then calling the instance for batch processing

COMMAND LINE USE: the batch process defined in the module can be run directly
from command line via (e.g.,) `python -m vtrack.batch [options]`. This 
basically instantize the appropriate batch processing class and then calls it 
with the supplied command line inputs. For the mapping of command line inputs 
to underlying function arguments, use the -h (or --help) option in command line

NOTE that in command line use the post-processing specifications in the
configuration is automatically converted into actual post-processing pipeline,
while such conversion has to be done manually when the batch processor is 
instantized normally. (The automatic conversion also happens when the batch
processor is used as a backend of the graphical user interface defined in 
vtrack.gui)
'''

from .core import *

import pathlib, re, json, itertools, abc

import imageio
from scipy.spatial import ConvexHull as _ConvexHull
from numpy import isnan as _isnan

def _fold_start_stop(start, stop, step, inner_dict):
    '''
    utility function to fold two sets of start/stop values in situation 
    involving .islice() iteration over another iterator constructed with
    .islice()
    
    ARGUMENTS:
      start: the start value of .islice() in the iterator being called
      stop: the stop value of .islice() in the iterator being called
      step: the step value of .islice() in the iterator being called
      inner_dict: dictionary containing the start, stop, step values of the
        .islice() in the calling function
        
    RETURNS:
      a 2-tuple of dictionary: first dictionary contains the updated 
      start, stop value for the .islice() in the function being called, and
      the second a copy of inner_dict with the start, stop values removed
    '''
    # copy to avoid modifying original inner_dict
    inner_dict = inner_dict.copy()
    
    # modify stop value
    stop_in = inner_dict.pop(stop, None)
    if stop_in is not None:
        stop = start + step * stop_in if (stop is None) else min(
            start + step * stop_in, stop
        )
    
    # modify start value
    start = start + step * inner_dict.pop("start", 0)
    
    # packing output dictionary
    outer_dict = {"start": start, "stop": stop}
    
    return (outer_dict, inner_dict)

def _flattened_yield(in_sequence):
    '''
    given an arbitrarily nested iterable, yield in the order of the 
    corresponding flattened structure
    
    ARGUMENTS: 
      in_sequence: an arbitrarily nested iterable
      
    YIELDS value from the corresponding flattened structure
    
    NOTE: for the purpose of this function, string (and its subclasses) are
    treated as atomic and NOT iterable
    '''
    try:
        iter(in_sequence)
    except TypeError:
        yield in_sequence
    else:
        if isinstance(in_sequence, str):
            yield in_sequence
        else:
            for item in in_sequence:
                yield from _flattened_yield(item)

def _flattened_list(in_sequence):
    '''
    convert an arbitrarily nested iterable into a flattened python list
    
    ARGUMENTS: 
      in_sequence: an arbitrarily nested iterable
      
    RETURNS:
        a flattened python list (with each element a shadow copy from the 
        corresponding element in in_sequence)
        
    NOTES: 
      1/ for the purpose of this function, string (and its subclasses) are
        treated as atomic and NOT iterable
      2/ in the special case where in_sequence is not a sequence, the 
        flattened output will be a list of a single element
    '''
    return list(_flattened_yield(in_sequence))

class ObjBatchBase(abc.ABC):
    '''
    Abstract base class for batch processing classes
    NOTE: has CONCRETE methods nonetheless!
    '''
    
    defaults = dict()
    
    def __init__(self):
        
        if config_file is None:
            self.copy_defaults()
        else:
            self.load_config(config_file)
        
        self.ops = []
        self.params = []
        self.kwparams = []
    
    def load_config(self, config_file, *, partial=False, **kwargs):
        '''
        load the given json file as config (previous config is discarded)
        
        ARGUMENTS:
          config_file: the name of the file that contains the json stings 
            from which the configuration is parsed from
          partial: if True, existing entries in current configuration that 
            are absent in config_file are left intact
          (**kwargs): the remaining keyword arguments are passed to json.load()
            that load and parse the json strings
        '''
        with open(config_file, "r") as infile:
            new_config = json.load(infile, **kwargs)
            
        if partial:
            for key, subdict in new_config:
                if key in self.config:
                    self.config[key].update(subdict)
                else:
                    self.config[key] = subdict
        else:
            self.config = new_config
    
    def copy_defaults(self):
        '''
        copy the class-level defined defaults as the configuration of the 
        current instance
        
        NO ARUGMENTS
        '''
        self.config = dict()
        for section, options in self.defaults.items():
            self.config[section] = options.copy()
    
    def save_config(self, config_file, **kwargs):
        '''
        Save the current configuation into json file
        
        ARGUMENTS:
          config_file: file name for which the json strings is written to
          (**kwargs): the remaining keyword arguments are passed to json.dump()
            that serialize and write the json strings
        '''
    
        with open(config_file, "w") as outfile:
            json.dump(self.config, outfile, **kwargs)
    
    def get_option(self, section, option):
        '''
        get the current value of a section or an option in the configuration
        
        ARGUMENTS:
          section: the section for which the option (or options, see below) 
            of interest resides in
          option: the option of interest. If None (a copy of) the entire 
            section is returned
        
        RETURNS:
          if section is None, the names (keys) of existing sections and the 
            options (keys) defined in each section, as a dict of list; 
            Otherwise, if option is not None, the value of the requested option
            from the specified section; And if option is None, a copy of 
            the dict from the specified section
        '''
        if section is None:
            return { 
                sect:list(sub_dict.keys()) for sect, sub_dict in 
                self.config.items()
            }
        if option is None:
            return self.config[section].copy()
        else:
            return self.config[section][option]
    
    def set_option(self, section, option, value, *, warn=False, stderr=print):
        '''
        set the value of a section or an option in the configuration
        
        ARGUMENTS:
          section: the section for which the option (or options, see below) 
            of interest resides in
          option: the option of interest. If None the entire section is 
            being set
          value: the value to be set (if option is None this should be a dict. 
            Note also that in such case the section is a COPY of the supplied 
            dict)
          warn: warn user if the option does not currently exists in the
            configuration (possible sign that the option may not be used)
          stderr: function that output the warning message to
        '''
        if option is None:
            if warn and (section not in self.config):
                stderr(
                    "WARNING: possibly irrelevant section {}".format(section)
                )
            self.config[section] = value.copy()
        else:
            if section not in self.config:
                if warn:
                    stderr(
                        "WARNING: possibly irrelevant " +
                        "option {} in section {}".format(section, option)
                    )
                self.config[section] = {option: value}
            else:
                if warn and (option not in self.config[section]):
                    stderr(
                        "WARNING: possibly irrelevant " +
                        "option {} in section {}".format(section, option)
                    )
                self.config[section][option] = value
    
    def insert_post_ops(self, func, params=None, kwparams=None, index=None):
        '''
        insert a new post-processing operation to the post-processing pipeline
        
        ARGUMENTS:
          func: the post-processing function to insert
          params: positional arguments associated with func, can be None
          kwparams: keyword arguments associated with func, can be None
          index: the index at which the post-processing function is to
            be inserted. if None inserting is at the end
        
        NOTES:
          Both params and kwparams are shallow copied
        '''
        if params is None:
            params = []
        if kwparams is None:
            kwparams = dict()
            
        if index is None:
            self.ops.append(func)
            self.params.append(params.copy())
            self.kwparams.append(kwparams.copy())
        
        else:
            self.ops.insert(i, func)
            self.params.insert(i, params.copy())
            self.kwparams.insert(i, kwparams.copy())
    
    def remove_post_ops(self, func):
        '''
        remove a post-processing operation from the post-processing pipeline
        
        ARGUMENTS:
          func: the function to be removed from the pipeline, OR the index
            at which the function is to be removed
        '''
        if callable(func):
            i = self.ops.index(func)
            self.ops.remove(func)
            self.params.pop(i)
            self.kwparams.pop(i)
        else:
            self.ops.pop(func)
            self.params.pop(func)
            self.kwparams.pop(func)
    
    def replace_post_ops(
        self, index, new_func=None, new_params=None, new_kwparams=None
    ):
        '''
        replace a post-processing operation in the post-processing pipeline
        
        ARGUMENTS:
          index: if integer, the index of the post-processing operation to 
            modify. If callable the function corresponding to the 
            post-processing operation to be modified
          new_func: the function to replace the original post-processing 
            operation. If None, the original function IS LEFT INTACT
          new_params: the new parameters to replace the original parameters 
            (positional arguments) of post-processing function. If None the
            original parameters are LEFT UNCHANGED
          new_kwparams: the new keyword parameters (keyword arguments) 
            to replace the original keyword parameters of post-processing 
            function. If None the original keyword parameters are UNCHANGED
            
        NOTES:
          Both new_params and new_kwparams are shallow copied
        '''
        if callable(index):
            index = self.ops.index(index)
        
        if new_func is not None:
            self.ops[index] = new_func
        
        if new_params is not None:
            self.params[index] = new_params.copy()
            
        if new_kwparams is not None:
            self.kwparams[index] = new_kwparams.copy()
    
    def clear_post_ops(self):
        '''
        remove all post-processing function from the post-processing pipeline
        
        NO ARGUMENTS
        '''
        self.ops.clear()
        self.params.clear()
        self.kwparams.clear()
    
    @abc.abstractmethod
    def infer_post_ops(self):
        pass
    
    @abc.abstractmethod
    def __call__(self):
        pass

class SingleObjBatch(ObjBatchBase):
    '''
    class for defining a "batch processor" for video motion analysis
    
    Basic usage: initialize a class instance by supplying a json formatted
    configuration file, then call the class instance with desired folders/
    files information to start batch processing
    
    When batch processing, the class instance load video files in succession,
    and perform the following pipeline of operations:
    
        calc_background -> [mark_image (bounding box)] -> calc_threshold -> 
        [mark_image (calibrate length)] -> postprocess -> compute_centroids -> 
        [mark_image (region of interest)] -> assemble_motion_data -> 
        write_csv -> plot_path -> export_overlaid_video
    
    where the operation in square brackets "[]" are performed only in 
    interaction mode, and some output may suppressed depending on settings.
    Also depending on settings the processor may output a summary file (of 
    total distance traveled and proportion in ROI) at the end
    
    The parsed configuration is structured as a dict of dict and can be
    accessed from the .config attribute of the instance. However, the 
    preferred way to access/manipulate the configuration is by using the 
    utility methods (.set_option(), .load_config(), etc.) defined for this
    class
    
    To describe the parsed configuration (henceforth "config" in short), we 
    shall refer to the keys of the outer dict as "sections" and the keys of 
    the inner dict as "options"
    
    The config consists of 4 mandatory sections termed "modes", "output", 
    "params", and "template_texts". As a general rule, all options within these
    4 sections should be supplied in the config file, EVEN IF the option is
    NOT being used (e.g., "figure_format" in "output" should be set even if 
    there is no figure to be outputted)
    
    The options under the "modes" section are:
      verbosity: (0, 1, or 2) how verbose is the messages printed to stdout 
        while the batch processing happens
      re: (bool) whether the patterns supplied when the class instance is 
        called are interpreted as regular expression or simple wildcard-
        enabled string
      eager: (bool) whether the image processing pipeline (up to 
        compute_centroids) is performed by maintaining a copy of video frames
        in memory (a.k.a. RAM). Otherwise the video will be read 3 separate 
        times (once of calc_background, calc_threshold, and compute_centroids).
        The "eager" mode is generally faster but MUCH MORE memory intensive,
        and should be used only for small (~1000 frames) videos
      smoothen: (bool) whether the motion data is smoothened
      interactive: (bool) whether the bounding box, length calibration, and
        region of interest is obtained interactively in EACH processed file
      overwrite: (bool) if true output files are overwritten if there is abs
        namely conflict. If not the output file is appended with "_#" (# an 
        integer) to avoid conflicts
    
    The options under the "output" section are:
      details: (bool) whether the detailed motion data is output to csv
      summary: (bool) whether the summary of motion data (one line for each
        file) is output to csv
      figure: (bool) whether to output (save as file) the main plot produced
        by plot_path() (plot_path() is skipped if this is false)
      video: (bool) whether to call export_overlaid_video() to generate an 
        overlaid video that will be saved
      record_threshold: (bool) if True, the threshold value will be saved to 
        the summary file
      record_conversion: (bool) if True, the pixel-to-physical unit conversion 
        factor will be asved to the summary file
      record_boxes: (bool) if True, the cropbox, maskbox, and innerbox 
        coordinates will be saved to the summary file
      figure_format: the format (file extension) for which the plot_path 
        figure will be saved in. Must INCLUDE the preceding dot (e.g., 
        ".png" not "png")
    
    Note that by default the summary csv always include: 
      1/ the input file name
      2/ the total distance the object traveled
      3/ the proportion of time the object spent in region of interest
    wherein 2 and/or 3 may be nan if appropriate
    
    There are 12 options under the "params" section, 7 of which has same 
    meaning whether in interaction mode or not:
      sample_interval: interval, in second, at which the frames from the video
        are sampled for computing background and luminosity thresholds (which 
        determines the "step" parameters in the respective function calls)
      step: step between successive frames to be extracted from the underlying
        video files
      start: the starting frame (count from 0) to be extracted from the 
        underlying video files
      stop: the stopping frame (exclusive; count from 0) for the underlying
        video files
      manual_threshold: a manually supplied threshold for luminosity. If you
        ant the threshold to be determined automatically set this to None 
        (null in json representation)
      convex: (bool) whether the bounding box and region of interests are 
        understood to be convex (if number of vertices > 2)
      window: window for motion smoothening. Must be an odd integer > 1
    In *non-interactive* mode the remaining options are interpreted as follow:
      crop_block: this parameter is ignored in non-interactive mode
      cropbox: specify cropbox in PIL convention (left, top, right, bottom).
        If None the images are not cropped
      maskbox: specify the mask-defining vertices in uncropped pixel 
        coordinates. If None no mask is applied
      innerbox: specify the vertices in physical (cropped and scaled) 
        coordinates for defining the region of interest. If None there is no 
        region of interest
      pixel_length: the conversion factor from pixel to physical unit
    While in *interactive* mode, these are interpreted as:
      crop_block: the minimum block size of cropbox. All dimensions of the
        cropbox must be divisible by crop_block
      cropbox: specify the number of vertices to be selected for defining 
        the cropbox. If None the images are not cropped
      maskbox: specify the number of vertices to be selected for defining 
        the mask. If None no mask is applied
      innerbox: specify the number of vertices to be selected for defining
        the region of interest. If None there is no region of interest
      pixel_length: the PHYSCIAL length corresponding to the segment selected
        in length calibration
    In addition, in interactive mode if both cropbox and maskbox takes the same
      non-None value, the two steps are merged into one where the cropbox is 
      automatically determined as the smallest rectangle enclosing the mask
    
    The "template_texts" section customizes messages displayed in prompt or 
    plot. These are template strings for which variables will be substituted 
    in (via the .format() method of python string) when actually used. Of the 
    7 options, the following 3 will be passed the name of input file and the
    number of input point requested (in that order):
      cropbox: Prompt to be displayed in selecting vertices for cropbox;
        interactive mode only
      maskbox: Prompt to be displayed in selecting vertices for maskbox;
        interactive mode only
      innerbox: Prompt to be displayed in selecting vertices for region of 
        interest; interactive mode only
    The following 2 options will be passed only the name of the input file:
      pixel_length: Prompt to be displayed in selecting vertices for length
        calibration; interactive mode only
      main_title: title of the main plot produced by plot_path()
    Finally, these 2 options will be passed the appropriate data:
      proportion: text to display in the main plot indicating proportion 
        time spent in region of interest
      distance: text to display in the main plot indicating total distance 
        traveled
    
    In addition to the 4 mandatory sections, the config may contain optional 
    sections whose inner dict are passed as keyword arguments to function calls
    in the pipeline, and whose name are the names of the corresponding function
    (e.g., customize the "plot_path" section to change the xlabel and ylabel 
    of the figure output). The functions with optional sections are 
    "calc_background", "calc_threshold", "mark_image", "assemble_motion_data", 
    "plot_path", and "export_overlaid_video"
    
    Finally, the config may also contain a "postprocess" section to succinctly
    specify common post-processing operations. The specification can then be
    turned into actual post-processing pipeline via the .inter_post_ops()
    method (without doing this the information in this section is simply
    ignored). This "postprocess" section has 2 recognized options:
      remove_spots: corresponds to post-processing frames using 
        skimage.morphology.remove_small_objects(). If value is boolean it
        specifies whether this post-processing is performed. If value is
        integer it specifies the smallest allowable object size
      remove_thins: corresponds to post-processing frames using 
        skimage.morphology.binary_opening(), which has the effect of removing
        thin foreground "filaments". If value is boolean it specifies whether
        this post-processing is performed. If value is integer it specifies
        the smallest allowable filament thickness (by setting the "selem"
        argument of binary_opening() to be skimage.morphology.square(value))
    NOTE that the order of operations is determined by the order at which
    the options are specified.
    
    Note that the conversion of specifications in the "postprocess" section
    into actual post-processing pipeline is automatically performed when the
    vtrack.batch module is used as a command line tool (via, e.g., python
    -m vtrack.batch) or when the class is used as a backend to the graphical
    user interface (GUI) created from vtrack.gui
    '''
    
    # class level default options
    defaults = {
        "modes": {
            "verbosity": 2,
            "re": False,
            "eager": False,
            "smoothen": True,
            "interactive": False,
            "overwrite": False
        },
        "output": {
            "details": True,
            "summary": True,
            "figure": True,
            "video": True,
            "record_threshold": True,
            "record_conversion": True,
            "record_boxes": True,
            "figure_format": ".png"
        },
        "params": {
            "sample_interval": 0.5,
            "step": 1,
            "start": 0,
            "stop": None,
            "manual_threshold": None,
            "convex": True,
            "window": 5,
            "crop_block": 8,
            "pixel_length": 1.0,
            "cropbox": None,
            "maskbox": None,
            "innerbox": None
        },
        "template_texts": {
            "cropbox": "{}: select {} vertices for the cropbox",
            "maskbox": "{}: select the {} vertices of the mask",
            "innerbox": "{}: select the {} vertices of the region of interest",
            "pixel_length": "{}: select 2 points of pixel_length apart",
            "main_title": "{}",
            "proportion": "proportion of time in ROI = {:.3f}",
            "distance": "total distance traveled = {:.0f}"
        }
    }
    
    def __init__(
        self, config_file=None, 
        post_ops=None, post_params=None, post_kwparams=None
    ):
        '''
        initialize batch processor
        
        ARGUMENTS:
          config_file: the name of the file that contains the json stings 
            from which the configuration is parsed from. If None the defaults
            defined at the class-level is used
          post_ops: list of functions, corresponding post-processing operations
            to be performed; can be None
          post_params: list of list, corresponding to positional arguments
            to be passed to the corresponding post_ops; can be None
          post_kwparams: list of dict, corresponding to keyword arguments to
            be passed to the corresponding post_ops; can be None
        '''
        if config_file is None:
            self.copy_defaults()
        else:
            self.load_config(config_file)
            
        if post_ops is None:
            self.ops = []
        else:
            self.ops = [ __ for __ in post_ops ]
        
        if post_params is None:
            self.params = [ [] for __ in self.ops ]
        else:
            self.params = [ __ for __ in post_params ]
            
        if post_kwparams is None:
            self.kwparams = [ dict() for __ in self.ops ]
        else:
            self.kwparams = [ __ for __ in post_kwparams ]
    
    @staticmethod
    def _glob_generator(in_folder, out_folder, glob, fig_suffix):
        "iterate through input file and output name in non-re mode"
        glob = "*" if (glob is None) else glob
        for in_path in pathlib.Path(in_folder).glob(glob):
            out_path = pathlib.Path(out_folder) / in_path.name
            csv_path = out_path.with_suffix(".csv")
            fig_path = out_path.with_suffix(fig_suffix)
            vid_path = out_path.with_suffix(".mp4")
            yield (in_path, csv_path, fig_path, vid_path)
    
    @staticmethod
    def _re_generator(
        in_folder, out_folder, in_pattern, out_pattern, fig_suffix
    ):
        "iterate through input file and output name in re mode"
        in_pattern = r".*" if (in_pattern is None) else in_pattern
        
        if out_pattern is None:
            out_pattern = [ r"\g<0>" ] * 3
        elif type(out_pattern)==str:
            out_pattern = [ out_pattern ] * 3
        
        in_rgx = re.compile(in_pattern)
        
        for in_path in pathlib.Path(in_folder).iterdir():
            if in_rgx.match(in_path.name):
                
                csv_path = pathlib.Path(out_folder) / in_rgx.sub(
                    out_pattern[0], in_path.name
                )
                csv_path = csv_path.with_suffix(".csv")
                
                fig_path = pathlib.Path(out_folder) / in_rgx.sub(
                    out_pattern[1], in_path.name
                )
                fig_path = fig_path.with_suffix(fig_suffix)
                
                vid_path = pathlib.Path(out_folder) / in_rgx.sub(
                    out_pattern[2], in_path.name
                )
                vid_path = vid_path.with_suffix(".mp4")
                
                yield (in_path, csv_path, fig_path, vid_path)
    
    @staticmethod
    def _gen_paths(*file_paths, junction="~"):
        "generate filename that avoids conflict with existing files"
        
        file_paths_0 = [ pathlib.Path(__) for __ in file_paths ]
        file_paths = file_paths_0
        i = 0
        while any([ __.exists() for __ in file_paths ]):
            i += 1
            file_paths = [ 
                __.with_name(__.stem + junction + str(i) + __.suffix)
                for __ in file_paths_0
            ]
        
        return tuple(file_paths)
    
    def infer_post_ops(self, *, clear=True):
        '''
        infer post-processing operations to perform based on the "postprocess" 
        section of the configuration
        
        ARGUMENTS:
          clear: whether to clear the existing post-processing pipeline
          
        RETURNS None
        
        SIDE EFFECTS: the post-processing pipeline is modified in-place
        '''
        
        # clear the existing postprocessing pipeline if instructed
        if clear:
            self.clear_post_ops()
        
        # nothing to do!
        if "postprocess" not in self.config:
            return
        
        # iterate through dictionary; order matters!
        for key, val in self.config["postprocess"].items():
            
            # hook for remove_small_objects()
            if key=="remove_spots":
                if type(val)==int: # min_size is specified
                    self.insert_post_ops(morph.remove_small_objects, [ val ])
                elif val: # use default
                    self.insert_post_ops(morph.remove_small_objects)
            
            # hook for binary_opening()
            if key=="remove_thins":
                if type(val)==int: # thickness is specified
                    self.insert_post_ops(
                        morph.binary_opening, [ morph.square(val) ]
                    )
                elif val: # use default otherwise
                    self.insert_post_ops(morph.binary_opening)
    
    def _run(self, filename_iterator, summary_path, *, stdout=print):
        '''
        execute batch processing
        
        ARGUMENTS:
          filename_iterator: an iterator that yields triplet of the form 
            (in_path, csv_path, fig_path, vid_path), where:
            - in_path is the path of the input file
            - csv_path is the path of the csv_file
            - fig_path is the path of the figure
            - vid_path is the path of the (output) video
            all 4 paths may be relative or absolute. Note that csv_path and
            fig_path, and vid_path are mandatory even if not written to
          summary_path: path of the summary file. Not used unless the 
            "summary" option is true in the "output" section of config
          stdout: function to write verbose messages to
        
        RETURNS None
    
        SIDE EFFECTS:
          files output to out_folder
        '''
        
        verbose = self.config["modes"]["verbosity"]
        
        # copy config into local alias; fill in missing sections if needed
        eager = self.config["modes"]["eager"]
        smooth = self.config["modes"]["smoothen"]
        interact = self.config["modes"]["interactive"]
        overwrite = self.config["modes"]["overwrite"]
        
        make_fig = self.config["output"]["figure"]
        make_csv = self.config["output"]["details"]
        make_vid = self.config["output"]["video"]
        make_smry = self.config["output"]["summary"]
        rec_thres = self.config["output"]["record_threshold"]
        rec_conv = self.config["output"]["record_conversion"]
        rec_boxes = self.config["output"]["record_boxes"]
        
        start = self.config["params"]["start"]
        stop = self.config["params"]["stop"]
        step = self.config["params"]["step"]
        convex = self.config["params"]["convex"]
        window = self.config["params"]["window"]
        man_thres = self.config["params"]["manual_threshold"]
        intvl = self.config["params"]["sample_interval"]
        
        if make_fig:
            title_template = self.config["template_texts"]["main_title"]
            time_template = self.config["template_texts"]["proportion"]
            dist_template = self.config["template_texts"]["distance"]
        
        bg_kwargs = self.config.get("calc_background", dict())
        th_kwargs = self.config.get("calc_threshold", dict())
        mot_kwargs = self.config.get("assemble_motion_data", dict())
        plt_kwargs = self.config.get("plot_path", dict())
        vid_kwargs = self.config.get("export_overlaid_video", dict())
        
        if not eager:
            bg_kwargs = _fold_start_stop(start, stop, step, bg_kwargs)
            if "step" in bg_kwargs[1]:
                bg_kwargs[0]["step"] = bg_kwargs[1].pop("step")
            th_kwargs = _fold_start_stop(start, stop, step, th_kwargs)
            if "step" in th_kwargs[1]:
                th_kwargs[0]["step"] = th_kwargs[1].pop("step")
        
        # saving all summary information
        if make_smry:
            all_summary = []
        
        # setup variables whose meaning depends on interaction mode
        if interact:
            known_length = self.config["params"]["pixel_length"]
            n_crop = self.config["params"]["cropbox"]
            n_mask = self.config["params"]["maskbox"]
            n_in = self.config["params"]["innerbox"]
            crop_block = self.config["params"]["crop_block"]
            cropbox0 = None # initial cropbox
            cropbox = n_crop # handle None case once and for all
            maskbox = None
            mask = n_mask # handle None case once and for all
            inbox = n_in
            inbox_path = n_in # handle None case once and for all
            mark_kwargs = self.config.get("mark_image", dict())
            crop_template = self.config["template_texts"]["cropbox"]
            mask_template = self.config["template_texts"]["maskbox"]
            in_template = self.config["template_texts"]["innerbox"]
            len_template = self.config["template_texts"]["pixel_length"]
            # special routing for getting cropbox and maskbox in one go
            if (n_crop is not None) and (n_crop == n_mask):
                n_mask = None
                mask_crop = True
                crop_template = mask_template
            else:
                mask_crop = False
        else:
            px2real = self.config["params"]["pixel_length"]
            cropbox0 = self.config["params"]["cropbox"]
            cropbox = None
            maskbox = self.config["params"]["maskbox"]
            inbox = self.config["params"]["innerbox"]
            n_crop = 2
            n_mask = None if (maskbox is None) else len(maskbox)
            n_in = None if (inbox is None) else len(inbox)
            mask_crop = False
            if convex:
                if (maskbox is not None) and (len(maskbox) > 2):
                    hull = _ConvexHull(maskbox)
                    maskbox = [ maskbox[__] for __ in hull.vertices ]
                if (inbox is not None) and (len(inbox) > 2):
                    hull = _ConvexHull(inbox)
                    inbox = [ inbox[__] for __ in hull.vertices ]
            mask = None # handle None case once and for all
            inbox_path = None if (inbox is None) else (
                vertices_to_path(inbox) # inbox in physical coordinates
            )
        
        for (in_path, csv_path, fig_path, vid_path) in filename_iterator:
            
            if verbose: stdout("Processing file '{}'...".format(in_path.name))
            reader = imageio.get_reader(in_path)
            meta = reader.get_meta_data()
            fps = meta["fps"] / step  # effective fps
            sub_step = max(1, int(intvl * fps))
            
            if eager:
            
                if verbose > 1: stdout("  building grayscaled frames...")
                frames = build_grayscaled(reader, step, start, stop, cropbox0)
                
                if verbose > 1: stdout("  calculating background frame...")
                bg_frame = calc_background(
                    frames, sub_step, cropbox=None, **bg_kwargs
                )
                
                if interact:
                    if n_crop is not None:
                        crop_pts = mark_image(
                            bg_frame, n_crop, convex, 
                            title = crop_template.format(in_path.name, n_crop),
                            **mark_kwargs
                        )
                        cropbox = vertices_to_cropbox(crop_pts, crop_block)
                        bg_cropped = bg_frame[cropbox_to_slices(cropbox)]
                        if mask_crop:
                            # maskbox in pixel coordinates
                            maskbox = convert_to_pixel(crop_pts)
                            mask = vertices_to_mask(maskbox, cropbox=cropbox)
                    else:
                        bg_cropped = bg_frame
                else:
                    bg_cropped = bg_frame
                
                if verbose > 1: stdout("  building subtracted frames...")
                frames = build_subtracted(frames, bg_frame, cropbox=cropbox)
                
                if man_thres is None:
                    if verbose > 1: stdout("  calculating threshold...")
                    thres = calc_threshold(
                        frames, sub_step, cropbox=None, **th_kwargs
                    )
                else:
                    thres = man_thres
                
                # NOTE: maskbox is expressed in pixel coordinates
                if interact:
                    if n_mask is not None:
                        maskbox = mark_image(
                            bg_cropped, n_mask, convex, 
                            title = mask_template.format(in_path.name, n_mask),
                            **mark_kwargs
                        )
                        maskbox = convert_to_pixel(maskbox, offset=cropbox)
                        mask = vertices_to_mask(
                            maskbox, image=bg_frame, cropbox=cropbox
                        )
                elif (maskbox is not None) and (mask is None):
                    mask = vertices_to_mask(
                        maskbox, image=bg_frame, cropbox=cropbox0
                    )
                
                if verbose > 1: stdout("  building binary frames...")
                frames = build_thresholded(
                    frames, thres, cropbox=None, mask=mask
                )
                
                if self.ops: # post-processing exists
                    if verbose > 1: stdout("  post-processing frames...")
                    frames = build_postprocessed(
                        frames, self.ops, self.params, self.kwparams
                    )
                
                if interact:
                    segment = mark_image(
                        bg_cropped, 2, False, 
                        title = len_template.format(in_path.name), 
                        **mark_kwargs
                    )
                    px2real = calc_pixel_to_phys(segment, known_length)
                
                if verbose > 1: stdout("  computing centroids...")
                coords = compute_centroids(frames, px2real)
            
            else: # "lazy" mode
                
                if verbose > 1: stdout(
                    "  generating grayscaled frames to calculate background..."
                )
                bg_frame = calc_background(
                    iter_grayscaled(
                        reader, step * sub_step, cropbox=cropbox0, 
                        **bg_kwargs[0]
                    ), cropbox=None, **bg_kwargs[1]
                )
                
                if interact:
                    if n_crop is not None:
                        crop_pts = mark_image(
                            bg_frame, n_crop, convex, 
                            title = crop_template.format(in_path.name, n_crop),
                            **mark_kwargs
                        )
                        cropbox = vertices_to_cropbox(crop_pts, crop_block)
                        bg_cropped = bg_frame[cropbox_to_slices(cropbox)]
                        if mask_crop:
                            # maskbox in pixel coordinates
                            maskbox = convert_to_pixel(crop_pts)
                            mask = vertices_to_mask(maskbox, cropbox=cropbox)
                    else:
                        bg_cropped = bg_frame
                else:
                    cropbox = cropbox0
                    bg_cropped = bg_frame
                
                if man_thres is None:
                    if verbose > 1: stdout(
                        "  generating subtracted frames " + 
                        "to calculate threshold..."
                    )
                    thres = calc_threshold(
                        iter_subtracted(
                            reader, bg_cropped, step * sub_step, 
                            cropbox=cropbox, **th_kwargs[0]
                        ), cropbox=None, **th_kwargs[1]
                    )
                else:
                    thres = man_thres
                
                # NOTE: maskbox is expressed in pixel coordinates
                if interact:
                    if n_mask is not None:
                        maskbox = mark_image(
                            bg_cropped, n_mask, convex, 
                            title = mask_template.format(in_path.name, n_mask),
                            **mark_kwargs
                        )
                        maskbox = convert_to_pixel(maskbox, offset=cropbox)
                        mask = vertices_to_mask(
                            maskbox, image=bg_frame, cropbox=cropbox
                        )
                elif (maskbox is not None) and (mask is None):
                    mask = vertices_to_mask(
                        maskbox, image=bg_frame, cropbox=cropbox0
                    )
                
                if interact:
                    segment = mark_image(
                        bg_cropped, 2, False, 
                        title = len_template.format(in_path.name), 
                        **mark_kwargs
                    )
                    px2real = calc_pixel_to_phys(segment, known_length)
                
                if self.ops: # post-processing exists
                    
                    if verbose > 1: stdout(
                        "  generating post-processed frames" + 
                        " to compute centroids..."
                    )
                    coords = compute_centroids(
                        iter_postprocessed(
                            reader, bg_cropped, thres, 
                            self.ops, self.params, self.kwparams,
                            step, start, stop,
                            cropbox=cropbox, mask=mask
                        ), px2real
                    )
                
                else: # no post-processing needed
                    
                    if verbose > 1: stdout(
                        "  generating binary frames to compute centroids..."
                    )
                    coords = compute_centroids(
                        iter_thresholded(
                            reader, bg_cropped, thres, step, start, stop,
                            cropbox=cropbox, mask=mask
                        ), px2real
                    )
                
            # eager and lazy modes merge back here
            
            if interact and (n_in is not None):
                inbox = mark_image(
                    bg_cropped, n_in, convex, 
                    title = in_template.format(in_path.name, n_in),
                    **mark_kwargs
                )
                # inbox in physical coordinates
                inbox = convert_to_physical(inbox, px2real)
                inbox_path = vertices_to_path(inbox)
            
            if verbose > 1: stdout("  computing motion data...")   
            dt = 1/fps
            data, header, summary = assemble_motion_data(
                coords, dt, marked_region=inbox_path, smooth=smooth,
                window=window, summary=True, **mot_kwargs
            )
            
            if verbose > 1: stdout("  generating output...")
            
            if not overwrite:
                paths = []
                if make_csv: paths.append(csv_path)
                if make_fig: paths.append(fig_path)
                if make_vid: paths.append(vid_path)
                paths = list(self._gen_paths(*paths))
                if make_vid: vid_path = paths.pop()
                if make_fig: fig_path = paths.pop()
                if make_csv: csv_path = paths.pop()
            
            if make_csv:
                write_csv(csv_path, data, header)
            
            if make_fig:
                info_list = []
                if not _isnan(summary.proportion):
                    info_list.append(time_template.format(summary.proportion))
                if not _isnan(summary.distance):
                    info_list.append(dist_template.format(summary.distance))
                plot_path(
                    title_template.format(in_path.name), data[:, 1:3], 
                    (data[0,0], data[-1,0]), inbox_path, 
                    info_list=info_list, save_as=fig_path, close=True, 
                    **plt_kwargs
                )
            
            if make_vid:
                box = cropbox if interact else cropbox0
                box = (0, 0) if (box is None) else box
                export_overlaid_video(
                    vid_path, itertools.islice(reader, start, stop, step), 
                    data[:, 1:3], px2real, box, 
                    fps=fps/step, **vid_kwargs
                )
            
            if make_smry:
                out_row = [ in_path.name ]
                out_row.extend(summary)
                if rec_thres: out_row.append(thres)
                if rec_conv: out_row.append(px2real)
                if rec_boxes: 
                    
                    # cropbox
                    if interact:
                        tmp = _flattened_list(cropbox)
                    else:
                        tmp = _flattened_list(cropbox0)
                    out_row.extend(tmp)
                    
                    # maskbox
                    tmp = _flattened_list(maskbox)
                    out_row.extend(tmp)
                    if mask_crop and (n_crop is not None):
                        out_row.extend(
                            ["" for __ in range(2 * n_crop - len(tmp))]
                        )
                    elif n_mask is not None:
                        out_row.extend(
                            ["" for __ in range(2 * n_mask - len(tmp))]
                        )
                    
                    # innerbox
                    tmp = _flattened_list(inbox)
                    out_row.extend(tmp)
                    if n_in is not None:
                        out_row.extend(
                            ["" for __ in range(2 * n_in - len(tmp))]
                        )
                    
                all_summary.append(out_row)
        
        ## END OF FILE ITERATION
        
        if make_smry:
            if verbose > 1: stdout("writing summary file...")
            if not overwrite:
                 summary_path, *_ = self._gen_paths(summary_path)
                 
            # create proper header
            header = ["file name", "total distance", "in-region time"]
            if rec_thres:
                header.append("threshold")
            if rec_conv:
                header.append("unit conversion")
            if rec_boxes:
            
                # cropbox
                tmp = cropbox if interact else cropbox0
                if tmp is None:
                    header.append("cropbox")
                else:
                    header.extend(["cropbox", "", "", ""])
                
                # maskbox
                header.append("maskbox")
                if mask_crop and (n_crop is not None):
                    header.extend(["" for __ in range(2 * n_crop - 1)])
                elif n_mask is not None:
                    header.extend(["" for __ in range(2 * n_mask - 1)])
                
                # innerbox
                header.append("innerbox")
                if n_in is not None:
                    header.extend(["" for __ in range(2 * n_in - 1)])
            
            write_csv(summary_path, all_summary, header)
        
        return
    
    def __call__(
        self, in_folder, out_folder, in_pattern=None, out_pattern=None,
        summary_name="summary.csv", *, stdout=print
    ):
        '''
        execute batch processing (via routing to internal _run() method)
        
        ARGUMENTS:
          in_folder: the folder for which the input videos are found
          out_folder: the folder for which the output files are written to
          in_pattern: in "non-re" mode, simple wild-card enabled string that
            specifies the input file (e.g., "*.mp4"). In "re" mode, a python
            recognized regular expression string that specifies the input file 
            (e.g., r"video_([0-9]{3})\\.mp4"). If None, defaults to all files in
            in_folder
          out_pattern: not used in "non-re" mode (wherein the output files have
            the same names as input file, but with file extension changed). In 
            "re" mode the replacement pattern that will be substituted into the
            input filename to produce the output file name (e.g., r"output\1" 
            produces "output123.csv" from "video_123.mp4" in the above 
            in_pattern). Note that file extension is optional and will always
            be replaced with the appropriate extensions. If neither None or
            a string, out_pattern should be sequence of string, and 
            substitution will be made separately for csv, figure, and video, 
            using the substitution patterns in that order.
          summary_name: name of the summary file. Not used unless the 
            "summary" option is true in the "output" section of config
          stdout: function to write verbose messages to
        
        RETURNS None
        
        SIDE EFFECTS:
          files output to out_folder
        '''
        
        # check if the in_folder and out_folder are actual directories
        if not pathlib.Path(in_folder).is_dir():
            raise FileNotFoundError("'{}' is not a folder".format(in_folder))
        if not pathlib.Path(out_folder).is_dir():
            raise FileNotFoundError("'{}' is not a folder".format(out_folder))
        
        if self.config["modes"]["verbosity"] > 1:
            stdout("Setting up main loop...")
        
        # determine mode to interpret __call__() input
        if self.config["modes"]["re"]:
            generator = self._re_generator(
                in_folder, out_folder, in_pattern, out_pattern,
                self.config["output"]["figure_format"]
            )
        else:
            generator = self._glob_generator(
                in_folder, out_folder, in_pattern,
                self.config["output"]["figure_format"]
            )
        
        # determine the full path of summary file
        smry_path = pathlib.Path(summary_name).with_suffix(".csv")
        if not smry_path.is_absolute():
            smry_path = pathlib.Path(out_folder) / smry_path
        
        # call internal _run() method for the real work
        self._run(generator, smry_path, stdout=stdout)
        
        if self.config["modes"]["verbosity"]:
            stdout("ALL DONE!")
        
        return

class MultiObjBatch(ObjBatchBase):
    '''
    class for defining a "batch processor" for video motion analysis
    
    Basic usage: initialize a class instance by supplying a json formatted
    configuration file, then call the class instance with desired folders/
    files information to start batch processing
    
    When batch processing, the class instance load video files in succession,
    and perform the following pipeline of operations:
    
        calc_background -> [mark_image (bounding box)] -> calc_threshold -> 
        [mark_image (calibrate length)] -> postprocess -> 
        compute_n_centroids -> [mark_image (region of interest)] -> 
        assemble_motion_data -> write_csv -> plot_n_paths -> 
        export_overlaid_n_video
    
    where the operation in square brackets "[]" are performed only in 
    interaction mode, and some output may suppressed depending on settings.
    Also depending on settings the processor may output a summary file (of 
    total distance traveled and proportion in ROI) at the end
    
    The parsed configuration is structured as a dict of dict and can be
    accessed from the .config attribute of the instance. However, the 
    preferred way to access/manipulate the configuration is by using the 
    utility methods (.set_option(), .load_config(), etc.) defined for this
    class
    
    To describe the parsed configuration (henceforth "config" in short), we 
    shall refer to the keys of the outer dict as "sections" and the keys of 
    the inner dict as "options"
    
    The config consists of 4 mandatory sections termed "modes", "output", 
    "params", and "template_texts". As a general rule, all options within these
    4 sections should be supplied in the config file, EVEN IF the option is
    NOT being used (e.g., "figure_format" in "output" should be set even if 
    there is no figure to be outputted)
    
    The options under the "modes" section are:
      verbosity: (0, 1, or 2) how verbose is the messages printed to stdout 
        while the batch processing happens
      re: (bool) whether the patterns supplied when the class instance is 
        called are interpreted as regular expression or simple wildcard-
        enabled string
      eager: (bool) whether the image processing pipeline (up to 
        compute_centroids) is performed by maintaining a copy of video frames
        in memory (a.k.a. RAM). Otherwise the video will be read 3 separate 
        times (once of calc_background, calc_threshold, and compute_centroids).
        The "eager" mode is generally faster but MUCH MORE memory intensive,
        and should be used only for small (~1000 frames) videos
      smoothen: (bool) whether the motion data is smoothened
      interactive: (bool) whether the bounding box, length calibration, and
        region of interest is obtained interactively in EACH processed file
      overwrite: (bool) if true output files are overwritten if there is abs
        namely conflict. If not the output file is appended with "_#" (# an 
        integer) to avoid conflicts
    
    The options under the "output" section are:
      details: (bool) whether the detailed motion data is output to csv
      summary: (bool) whether the summary of motion data (one line for each
        file) is output to csv
      figure: (bool) whether to output (save as file) the main plot produced
        by plot_path() (plot_path() is skipped if this is false)
      video: (bool) whether to call export_overlaid_video() to generate an 
        overlaid video that will be saved
      record_threshold: (bool) if True, the threshold value will be saved to 
        the summary file
      record_conversion: (bool) if True, the pixel-to-physical unit conversion 
        factor will be asved to the summary file
      record_boxes: (bool) if True, the cropbox, maskbox, and innerbox 
        coordinates will be saved to the summary file
      figure_format: the format (file extension) for which the plot_path 
        figure will be saved in. Must INCLUDE the preceding dot (e.g., 
        ".png" not "png")
      multi_figures: (bool) if True, each object is plotted in a separate 
        figure using plot_path(). If False, all objects are plotted 
        on the same figure using plot_n_paths()
      figures_naming: template string used for naming figures if multi_figuares
        is True (ignored otherwise). The template string is passed the figure
        name generated by input name (after out_pattern substitution if 
        applicable) and the object index (starting from 0)
    
    Note that by default the summary csv always include: 
      1/ the input file name
      2/ the total distance the object traveled
      3/ the proportion of time the object spent in region of interest
    wherein 2 and/or 3 may be nan if appropriate
    
    There are 14 options under the "params" section, 9 of which has same 
    meaning whether in interaction mode or not:
      n_obj: the number of foreground objects to detect in the video
      sample_interval: interval, in second, at which the frames from the video
        are sampled for computing background and luminosity thresholds (which 
        determines the "step" parameters in the respective function calls)
      step: step between successive frames to be extracted from the underlying
        video files
      start: the starting frame (count from 0) to be extracted from the 
        underlying video files
      stop: the stopping frame (exclusive; count from 0) for the underlying
        video files
      manual_threshold: a manually supplied threshold for luminosity. If you
        ant the threshold to be determined automatically set this to None 
        (null in json representation)
      convex: (bool) whether the bounding box and region of interests are 
        understood to be convex (if number of vertices > 2)
      reinitialize: whether to reinitialize the inital guess for centroids 
        for each frame (if True), or to just use the centroid from the 
        previous frame as initial guess (if False)
      window: window for motion smoothening. Must be an odd integer > 1
    In *non-interactive* mode the remaining options are interpreted as follow:
      crop_block: this parameter is ignored in non-interactive mode
      cropbox: specify cropbox in PIL convention (left, top, right, bottom).
        If None the images are not cropped
      maskbox: specify the mask-defining vertices in uncropped pixel 
        coordinates. If None no mask is applied
      innerbox: specify the vertices in physical (cropped and scaled) 
        coordinates for defining the region of interest. If None there is no 
        region of interest
      pixel_length: the conversion factor from pixel to physical unit
    While in *interactive* mode, these are interpreted as:
      crop_block: the minimum block size of cropbox. All dimensions of the
        cropbox must be divisible by crop_block
      cropbox: specify the number of vertices to be selected for defining 
        the cropbox. If None the images are not cropped
      maskbox: specify the number of vertices to be selected for defining 
        the mask. If None no mask is applied
      innerbox: specify the number of vertices to be selected for defining
        the region of interest. If None there is no region of interest
      pixel_length: the PHYSCIAL length corresponding to the segment selected
        in length calibration
    In addition, in the interactive mode if both cropbox and maskbox takes the
      same non-None value, the two steps are merged into one where the cropbox
      is automatically determined to be the smallest cropbox enclosing the mask
    
    The "template_texts" section customizes messages displayed in prompt or 
    plot. These are template strings for which variables will be substituted 
    in (via the .format() method of python string) when actually used. Of the 
    7 options, the following 3 will be passed the name of input file and the
    number of input point requested (in that order):
      cropbox: Prompt to be displayed in selecting vertices for cropbox;
        interactive mode only
      maskbox: Prompt to be displayed in selecting vertices for maskbox;
        interactive mode only
      innerbox: Prompt to be displayed in selecting vertices for region of 
        interest; interactive mode only
    The following option will be passed the name of the input file and the
    object index (starting from 1) if "multi_figures" in "output" section is
    True, and only the name of the input file if "multi_figures" is False:
      main_title: title of the main plot produced by plot_path() or 
        plot_n_paths()
    The following option will be passed only the name of the input file:
      pixel_length: Prompt to be displayed in selecting vertices for length
        calibration; interactive mode only
    Finally, these 2 options will be passed the appropriate data:
      proportion: text to display in the main plot indicating proportion 
        time spent in region of interest
      distance: text to display in the main plot indicating total distance 
        traveled
    
    In addition to the 4 mandatory sections, the config may contain optional 
    sections whose inner dict are passed as keyword arguments to function calls
    in the pipeline, and whose name are the names of the corresponding function
    (e.g., customize the "plot_n_paths" section to change the xlabel and ylabel 
    of the figure output). The functions with optional sections are 
    "calc_background", "calc_threshold", "mark_image", "compute_n_centroids",
    "assemble_motion_data", "plot_path", "plot_n_paths", and 
    "export_overlaid_n_video"
    
    Finally, the config may also contain a "postprocess" section to succinctly
    specify common post-processing operations. The specification can then be
    turned into actual post-processing pipeline via the .inter_post_ops() 
    method (without doing this the information in this section is simply 
    ignored). This "postprocess" section has 4 recognized options:
      remove_spots: corresponds to post-processing frames using 
        skimage.morphology.remove_small_objects(). If value is boolean it
        specifies whether this post-processing is performed. If value is
        integer it specifies the smallest allowable object size
      remove_thins: corresponds to post-processing frames using 
        skimage.morphology.binary_opening(), which has the effect of removing
        thin foreground "filaments". If value is boolean it specifies whether 
        this post-processing is performed. If value is integer it specifies 
        the smallest allowable filament thickness (by setting the "selem" 
        argument of binary_opening() to be skimage.morphology.square(value))
    NOTE that the order of operations is determined by the order at which
    the options are specified.
    
    Note that the conversion of specifications in the "postprocess" section
    into actual postprocessing pipeline is automatically performed when the
    vtrack.batch module is used as a command line tool (via, e.g., python 
    -m vtrack.batch) or when the class is used as a backend to the graphical 
    user interface (GUI) created from vtrack.gui
    '''
    
    # class level default options
    defaults = {
        "modes": {
            "verbosity": 2,
            "re": False,
            "eager": False,
            "smoothen": True,
            "interactive": False,
            "overwrite": False
        },
        "output": {
            "details": True,
            "summary": True,
            "figure": True,
            "video": True,
            "record_threshold": True,
            "record_conversion": True,
            "record_boxes": True,
            "figure_format": ".png",
            "multi_figures": False,
            "figures_naming": "{}_{}"
        },
        "params": {
            "n_obj": 2,
            "sample_interval": 0.5,
            "step": 1,
            "start": 0,
            "stop": None,
            "manual_threshold": None,
            "convex": True,
            "reinitialize": False,
            "window": 5,
            "crop_block": 8,
            "pixel_length": 1.0,
            "cropbox": None,
            "maskbox": None,
            "innerbox": None
        },
        "template_texts": {
            "cropbox": "{}: select {} vertices for the cropbox",
            "maskbox": "{}: select the {} vertices of the mask",
            "innerbox": "{}: select the {} vertices of the region of interest",
            "pixel_length": "{}: select 2 points of pixel_length apart",
            "main_title": "{}",
            "proportion": "object #{}: proportion of time in ROI = {:.3f}",
            "distance": "object #{}: total distance traveled = {:.0f}"
        }
    }
    
    def __init__(
        self, config_file=None, 
        post_ops=None, post_params=None, post_kwparams=None
    ):
        '''
        initialize batch processor
        
        ARGUMENTS:
          config_file: the name of the file that contains the json stings 
            from which the configuration is parsed from. If None the defaults
            defined at the class-level is used
          post_ops: list of functions, corresponding to a sequence of 
            post-processing operations to be performed; can be None
          post_params: list of list, corresponding to positional arguments
            to be passed to the corresponding post_ops; can be None
          post_kwparams: list of dict, corresponding to keyword arguments to
            be passed to the corresponding post_ops; can be None
        '''
        if config_file is None:
            self.copy_defaults()
        else:
            self.load_config(config_file)
            
        if post_ops is None:
            self.ops = []
        else:
            self.ops = [ __ for __ in post_ops ]
        
        if post_params is None:
            self.params = [ [] for __ in self.ops ]
        else:
            self.params = [ __ for __ in post_params ]
            
        if post_kwparams is None:
            self.kwparams = [ dict() for __ in self.ops ]
        else:
            self.kwparams = [ __ for __ in post_kwparams ]
    
    @staticmethod
    def _glob_generator(in_folder, out_folder, glob, fig_suffix):
        "iterate through input file and output name in non-re mode"
        glob = "*" if (glob is None) else glob
        for in_path in pathlib.Path(in_folder).glob(glob):
            out_path = pathlib.Path(out_folder) / in_path.name
            csv_path = out_path.with_suffix(".csv")
            fig_path = out_path.with_suffix(fig_suffix)
            vid_path = out_path.with_suffix(".mp4")
            yield (in_path, csv_path, fig_path, vid_path)
    
    @staticmethod
    def _re_generator(
        in_folder, out_folder, in_pattern, out_pattern, fig_suffix
    ):
        "iterate through input file and output name in re mode"
        in_pattern = r".*" if (in_pattern is None) else in_pattern
        
        if out_pattern is None:
            out_pattern = [ r"\g<0>" ] * 3
        elif type(out_pattern)==str:
            out_pattern = [ out_pattern ] * 3
        
        in_rgx = re.compile(in_pattern)
        
        for in_path in pathlib.Path(in_folder).iterdir():
            if in_rgx.match(in_path.name):
                
                csv_path = pathlib.Path(out_folder) / in_rgx.sub(
                    out_pattern[0], in_path.name
                )
                csv_path = csv_path.with_suffix(".csv")
                
                fig_path = pathlib.Path(out_folder) / in_rgx.sub(
                    out_pattern[1], in_path.name
                )
                fig_path = fig_path.with_suffix(fig_suffix)
                
                vid_path = pathlib.Path(out_folder) / in_rgx.sub(
                    out_pattern[2], in_path.name
                )
                vid_path = vid_path.with_suffix(".mp4")
                
                yield (in_path, csv_path, fig_path, vid_path)
    
    @staticmethod
    def _gen_paths(*file_paths, junction="~"):
        "generate filename that avoids conflict with existing files"
        
        file_paths_0 = [ pathlib.Path(__) for __ in file_paths ]
        file_paths = file_paths_0
        i = 0
        while any([ __.exists() for __ in file_paths ]):
            i += 1
            file_paths = [ 
                __.with_name(__.stem + junction + str(i) + __.suffix)
                for __ in file_paths_0
            ]
        
        return tuple(file_paths)
    
    def infer_post_ops(self, *, clear=True):
        '''
        infer post-processing operations to perform based on the "postprocess"
        section of the configuration
        
        ARGUMENTS:
          clear: whether to clear the existing post-processing pipeline
          
        RETURNS None
        
        SIDE EFFECTS: the post-processing pipeline is modified in-place
        '''
        
        # clear the existing postprocessing pipeline if instructed
        if clear:
            self.clear_post_ops()
        
        # nothing to do!
        if "postprocess" not in self.config:
            return
        
        postprocess = self.config["postprocess"]
        
        # remove_small_objects
        if postprocess.get("remove_spots", False):
            if "spot_size" in postprocess:
                self.insert_post_ops(
                    morph.remove_small_objects,
                    [ postprocess["spot_size"] ]
                )
            else:
                self.insert_post_ops(morph.remove_small_objects)
        
        # binary_opening
        if postprocess.get("remove_thins", False):
            if "thickness" in postprocess:
                self.insert_post_ops(
                    morph.binary_opening,
                    [ postprocess["thickness"] ]
                )
            else:
                self.insert_post_ops(morph.binary_opening)
    
    def _run(self, filename_iterator, summary_path, *, stdout=print):
        '''
        execute batch processing
        
        ARGUMENTS:
          filename_iterator: an iterator that yields triplet of the form 
            (in_path, csv_path, fig_path, vid_path), where:
            - in_path is the path of the input file
            - csv_path is the path of the csv_file
            - fig_path is the path of the figure
            - vid_path is the path of the (output) video
            all 4 paths may be relative or absolute. Note that csv_path and
            fig_path, and vid_path are mandatory even if not written to
          summary_path: path of the summary file. Not used unless the 
            "summary" option is true in the "output" section of config
          stdout: function to write verbose messages to
        
        RETURNS None
        
        SIDE EFFECTS:
          files output to out_folder
        '''
        
        verbose = self.config["modes"]["verbosity"]
        
        # copy config into local alias; fill in missing sections if needed
        eager = self.config["modes"]["eager"]
        smooth = self.config["modes"]["smoothen"]
        interact = self.config["modes"]["interactive"]
        overwrite = self.config["modes"]["overwrite"]
        
        make_fig = self.config["output"]["figure"]
        make_csv = self.config["output"]["details"]
        make_vid = self.config["output"]["video"]
        make_smry = self.config["output"]["summary"]
        rec_thres = self.config["output"]["record_threshold"]
        rec_conv = self.config["output"]["record_conversion"]
        rec_boxes = self.config["output"]["record_boxes"]
        multi_figs = self.config["output"]["multi_figures"]
        
        n_obj = self.config["params"]["n_obj"]
        start = self.config["params"]["start"]
        stop = self.config["params"]["stop"]
        step = self.config["params"]["step"]
        convex = self.config["params"]["convex"]
        reinit = self.config["params"]["reinitialize"]
        window = self.config["params"]["window"]
        man_thres = self.config["params"]["manual_threshold"]
        intvl = self.config["params"]["sample_interval"]
        
        if make_fig:
            title_template = self.config["template_texts"]["main_title"]
            time_template = self.config["template_texts"]["proportion"]
            dist_template = self.config["template_texts"]["distance"]
            if multi_figs:
                plt_kwargs = self.config.get("plot_path", dict())
                fig_template = self.config["output"]["figures_naming"]
                fig_fmt = self.config["output"]["figure_format"]
            else:
                plt_kwargs = self.config.get("plot_n_paths", dict())
        
        bg_kwargs = self.config.get("calc_background", dict())
        th_kwargs = self.config.get("calc_threshold", dict())
        cen_kwargs = self.config.get("compute_n_centroids", dict())
        mot_kwargs = self.config.get("assemble_motion_data", dict())

        vid_kwargs = self.config.get("export_overlaid_n_video", dict())
        
        if not eager:
            bg_kwargs = _fold_start_stop(start, stop, step, bg_kwargs)
            if "step" in bg_kwargs[1]:
                bg_kwargs[0]["step"] = bg_kwargs[1].pop("step")
            th_kwargs = _fold_start_stop(start, stop, step, th_kwargs)
            if "step" in th_kwargs[1]:
                th_kwargs[0]["step"] = th_kwargs[1].pop("step")
        
        # saving all summary information
        if make_smry:
            all_summary = []
        
        # setup variables whose meaning depends on interaction mode
        if interact:
            known_length = self.config["params"]["pixel_length"]
            n_crop = self.config["params"]["cropbox"]
            n_mask = self.config["params"]["maskbox"]
            n_in = self.config["params"]["innerbox"]
            crop_block = self.config["params"]["crop_block"]
            cropbox0 = None # initial cropbox
            cropbox = n_crop # handle None case once and for all
            maskbox = None
            mask = n_mask # handle None case once and for all
            inbox = n_in
            inbox_path = n_in # handle None case once and for all
            mark_kwargs = self.config.get("mark_image", dict())
            crop_template = self.config["template_texts"]["cropbox"]
            mask_template = self.config["template_texts"]["maskbox"]
            in_template = self.config["template_texts"]["innerbox"]
            len_template = self.config["template_texts"]["pixel_length"]
            # special routing for getting cropbox and maskbox in one go
            if (n_crop is not None) and (n_crop == n_mask):
                n_mask = None
                mask_crop = True
                crop_template = mask_template
            else:
                mask_crop = False
        else:
            px2real = self.config["params"]["pixel_length"]
            cropbox0 = self.config["params"]["cropbox"]
            cropbox = None
            maskbox = self.config["params"]["maskbox"]
            inbox = self.config["params"]["innerbox"]
            n_crop = 2
            n_mask = None if (maskbox is None) else len(maskbox)
            n_in = None if (inbox is None) else len(inbox)
            mask_crop = False
            if convex:
                if (maskbox is not None) and (len(maskbox) > 2):
                    hull = _ConvexHull(maskbox)
                    maskbox = [ maskbox[__] for __ in hull.vertices ]
                if (inbox is not None) and (len(inbox) > 2):
                    hull = _ConvexHull(inbox)
                    inbox = [ inbox[__] for __ in hull.vertices ]
            mask = None # handle None case once and for all
            inbox_path = None if (inbox is None) else (
                vertices_to_path(inbox, px2real)
            )
        
        for (in_path, csv_path, fig_path, vid_path) in filename_iterator:
            
            if verbose: stdout("Processing file '{}'...".format(in_path.name))
            reader = imageio.get_reader(in_path)
            meta = reader.get_meta_data()
            fps = meta["fps"] / step  # effective fps
            sub_step = max(1, int(intvl * fps))
            
            if eager:
            
                if verbose > 1: stdout("  building grayscaled frames...")
                frames = build_grayscaled(reader, step, start, stop, cropbox0)
                
                if verbose > 1: stdout("  calculating background frame...")
                bg_frame = calc_background(
                    frames, sub_step, cropbox=None, **bg_kwargs
                )
                
                if interact:
                    if n_crop is not None:
                        crop_pts = mark_image(
                            bg_frame, n_crop, convex, 
                            title = crop_template.format(in_path.name, n_crop),
                            **mark_kwargs
                        )
                        cropbox = vertices_to_cropbox(crop_pts, crop_block)
                        bg_cropped = bg_frame[cropbox_to_slices(cropbox)]
                        if mask_crop:
                            # maskbox in pixel coordinates
                            maskbox = convert_to_pixel(crop_pts)
                            mask = vertices_to_mask(maskbox, cropbox=cropbox)
                    else:
                        bg_cropped = bg_frame
                else:
                    bg_cropped = bg_frame
                
                if verbose > 1: stdout("  building subtracted frames...")
                frames = build_subtracted(frames, bg_frame, cropbox=cropbox)
                
                if man_thres is None:
                    if verbose > 1: stdout("  calculating threshold...")
                    thres = calc_threshold(
                        frames, sub_step, cropbox=None, **th_kwargs
                    )
                else:
                    thres = man_thres
                
                # NOTE: maskbox is expressed in pixel coordinates
                if interact:
                    if n_mask is not None:
                        maskbox = mark_image(
                            bg_cropped, n_mask, convex, 
                            title = mask_template.format(in_path.name, n_mask),
                            **mark_kwargs
                        )
                        maskbox = convert_to_pixel(maskbox, offset=cropbox)
                        mask = vertices_to_mask(
                            maskbox, image=bg_frame, cropbox=cropbox
                        )
                elif (maskbox is not None) and (mask is None):
                    mask = vertices_to_mask(
                        maskbox, image=bg_frame, cropbox=cropbox0
                    )
                
                if verbose > 1: stdout("  building binary frames...")
                frames = build_thresholded(
                    frames, thres, cropbox=None, mask=mask
                )
                
                if self.ops: # post-processing exists
                    if verbose > 1: stdout("  post-processing frames...")
                    frames = build_postprocessed(
                        frames, self.ops, self.params, self.kwparams
                    )
                
                if interact:
                    segment = mark_image(
                        bg_cropped, 2, False, 
                        title = len_template.format(in_path.name), 
                        **mark_kwargs
                    )
                    px2real = calc_pixel_to_phys(segment, known_length)
                
                if verbose > 1: stdout("  computing centroids...")
                coords = compute_n_centroids(
                    frames, n_obj, px2real, reinitialize=reinit, **cen_kwargs
                )
            
            else: # "lazy" mode
                
                if verbose > 1: stdout(
                    "  generating grayscaled frames to calculate background..."
                )
                bg_frame = calc_background(
                    iter_grayscaled(
                        reader, step * sub_step, cropbox=cropbox0, 
                        **bg_kwargs[0]
                    ), cropbox=None, **bg_kwargs[1]
                )
                
                if interact:
                    if n_crop is not None:
                        crop_pts = mark_image(
                            bg_frame, n_crop, convex, 
                            title = crop_template.format(in_path.name, n_crop),
                            **mark_kwargs
                        )
                        cropbox = vertices_to_cropbox(crop_pts, crop_block)
                        bg_cropped = bg_frame[cropbox_to_slices(cropbox)]
                        if mask_crop:
                            # maskbox in pixel coordinates
                            maskbox = convert_to_pixel(crop_pts)
                            mask = vertices_to_mask(maskbox, cropbox=cropbox)
                    else:
                        bg_cropped = bg_frame
                else:
                    cropbox = cropbox0
                    bg_cropped = bg_frame
                
                if man_thres is None:
                    if verbose > 1: stdout(
                        "  generating subtracted frames " + 
                        "to calculate threshold..."
                    )
                    thres = calc_threshold(
                        iter_subtracted(
                            reader, bg_cropped, step * sub_step, 
                            cropbox=cropbox, **th_kwargs[0]
                        ), cropbox=None, **th_kwargs[1]
                    )
                else:
                    thres = man_thres
                
                # NOTE: maskbox is expressed in pixel coordinates
                if interact:
                    if n_mask is not None:
                        maskbox = mark_image(
                            bg_cropped, n_mask, convex, 
                            title = mask_template.format(in_path.name, n_mask),
                            **mark_kwargs
                        )
                        maskbox = convert_to_pixel(maskbox, offset=cropbox)
                        mask = vertices_to_mask(maskbox, image=bg_cropped)
                elif (maskbox is not None) and (mask is None):
                    mask = vertices_to_mask(
                        maskbox, image=bg_frame, cropbox=cropbox0
                    )
                
                if interact:
                    segment = mark_image(
                        bg_cropped, 2, False, 
                        title = len_template.format(in_path.name), 
                        **mark_kwargs
                    )
                    px2real = calc_pixel_to_phys(segment, known_length)
                
                if self.ops: # post-processing exists
                    
                    if verbose > 1: stdout(
                        "  generating post-processed frames" + 
                        " to compute centroids..."
                    )
                    coords = compute_n_centroids(
                        iter_postprocessed(
                            reader, bg_cropped, thres, 
                            self.ops, self.params, self.kwparams,
                            step, start, stop,
                            cropbox=cropbox, mask=mask
                        ), n_obj, px2real, reinitialize=reinit, **cen_kwargs
                    )
                
                else: # no post-processing needed
                    
                    if verbose > 1: stdout(
                        "  generating binary frames to compute centroids..."
                    )
                    coords = compute_n_centroids(
                        iter_thresholded(
                            reader, bg_cropped, thres, step, start, stop,
                            cropbox=cropbox, mask=mask
                        ), n_obj, px2real, reinitialize=reinit, **cen_kwargs
                    )
                
            # eager and lazy modes merge back here
            
            if interact and (n_in is not None):
                inbox = mark_image(
                    bg_cropped, n_in, convex, 
                    title = in_template.format(in_path.name, n_in),
                    **mark_kwargs
                )
                # inbox in physical coordinates
                inbox = convert_to_physical(inbox, px2real)
                inbox_path = vertices_to_path(inbox)
            
            if verbose > 1: stdout("  computing motion data...")
            dt = 1/fps
            data, header, summaries = assemble_motion_data(
                coords, dt, marked_region=inbox_path, smooth=smooth,
                window=window, summary=True, **mot_kwargs
            )
            
            if verbose > 1: stdout("  generating output...")
            
            if multi_figs:
                fig_paths = []
                for i in range(1, n_obj + 1):
                    new_name = fig_template.format(fig_path.stem, i)
                    fig_paths.append(
                        fig_path.with_name(new_name).with_suffix(fig_fmt)
                    )
            
            if not overwrite:
                paths = []
                if make_csv: paths.append(csv_path)
                if make_fig: 
                    if multi_figs:
                        paths.extend(fig_paths)
                    else:
                        paths.append(fig_path)
                if make_vid: paths.append(vid_path)
                paths = list(self._gen_paths(*paths))
                if make_vid: vid_path = paths.pop()
                if make_fig: 
                    if multi_figs:
                        fig_paths = paths[-n_obj:]
                        paths = paths[:-n_obj]
                    else:
                        fig_path = paths.pop()
                if make_csv: csv_path = paths.pop()
            
            if make_csv:
                write_csv(
                    csv_path, 
                    data.reshape(data.shape[0], -1), 
                    _flattened_list(header)
                )
            
            if make_fig:
                if multi_figs:
                    for i in range(n_obj):
                        info_list = []
                        summary = summaries[i]
                        if not _isnan(summary.proportion):
                            info_list.append(
                                time_template.format(i, summary.proportion)
                            )
                        if not _isnan(summary.distance):
                            info_list.append(
                                dist_template.format(i, summary.distance)
                            )
                        plot_path(
                            title_template.format(in_path.name, i+1), 
                            data[:, i, 1:3], (data[0,i,0], data[-1,i,0]), 
                            inbox_path, info_list=info_list, 
                            save_as=fig_paths[i], close=True, 
                            **plt_kwargs
                        )
                else:
                    info_list = []
                    for i, summary in enumerate(summaries, start=1):
                        if not _isnan(summary.proportion):
                            info_list.append(
                                time_template.format(i, summary.proportion)
                            )
                        if not _isnan(summary.distance):
                            info_list.append(
                                dist_template.format(i, summary.distance)
                            )
                    plot_n_paths(
                        title_template.format(in_path.name), data[:, :, 1:3], 
                        (data[0,0,0], data[-1,0,0]), inbox_path, 
                        info_list=info_list, save_as=fig_path, close=True, 
                        **plt_kwargs
                    )
            
            if make_vid:
                box = cropbox if interact else cropbox0
                box = (0, 0) if (box is None) else box
                export_overlaid_n_video(
                    vid_path, itertools.islice(reader, start, stop, step), 
                    data[:, :, 1:3], px2real, box, 
                    fps=fps/step, **vid_kwargs
                )
            
            if make_smry:
                out_row = [ in_path.name ]
                for summary in summaries:
                    out_row.extend(summary)
                if rec_thres: out_row.append(thres)
                if rec_conv: out_row.append(px2real)
                if rec_boxes: 
                    
                    # cropbox
                    if interact:
                        tmp = _flattened_list(cropbox)
                    else:
                        tmp = _flattened_list(cropbox0)
                    out_row.extend(tmp)
                    
                    # maskbox
                    tmp = _flattened_list(maskbox)
                    out_row.extend(tmp)
                    if mask_crop and (n_crop is not None):
                        out_row.extend(
                            ["" for __ in range(2 * n_crop - len(tmp))]
                        )
                    elif n_mask is not None:
                        out_row.extend(
                            ["" for __ in range(2 * n_mask - len(tmp))]
                        )
                    
                    # innerbox
                    tmp = _flattened_list(inbox)
                    out_row.extend(tmp)
                    if n_in is not None:
                        out_row.extend(
                            ["" for __ in range(2 * n_in - len(tmp))]
                        )
                    
                all_summary.append(out_row)
        
        ## END OF FILE ITERATION
        
        if make_smry:
            if verbose > 1: stdout("writing summary file...")
            if not overwrite:
                 summary_path, *_ = self._gen_paths(summary_path)
                 
            # create proper header
            header = ["file name"]
            for i in range(1, n_obj + 1):
                header.extend([
                    "total distance #{}".format(i), 
                    "in-region time #{}".format(i)
                ])
            if rec_thres:
                header.append("threshold")
            if rec_conv:
                header.append("unit conversion")
            if rec_boxes:
            
                # cropbox
                tmp = cropbox if interact else cropbox0
                if tmp is None:
                    header.append("cropbox")
                else:
                    header.extend(["cropbox", "", "", ""])
                
                # maskbox
                header.append("maskbox")
                if mask_crop and (n_crop is not None):
                    header.extend(["" for __ in range(2 * n_crop - 1)])
                elif n_mask is not None:
                    header.extend(["" for __ in range(2 * n_mask - 1)])
                
                # innerbox
                header.append("innerbox")
                if n_in is not None:
                    header.extend(["" for __ in range(2 * n_in - 1)])
            
            write_csv(summary_path, all_summary, header)
        
        return
    
    def __call__(
        self, in_folder, out_folder, in_pattern=None, out_pattern=None,
        summary_name="summary.csv", *, stdout=print
    ):
        '''
        execute batch processing (via routing to internal _run() method)
        
        ARGUMENTS:
          in_folder: the folder for which the input videos are found
          out_folder: the folder for which the output files are written to
          in_pattern: in "non-re" mode, simple wild-card enabled string that
            specifies the input file (e.g., "*.mp4"). In "re" mode, a python
            recognized regular expression string that specifies the input file 
            (e.g., r"video_([0-9]{3})\\.mp4"). If None, defaults to all files in
            in_folder
          out_pattern: not used in "non-re" mode (wherein the output files have
            the same names as input file, but with file extension changed). In 
            "re" mode the replacement pattern that will be substituted into the
            input filename to produce the input file name (e.g., r"output\1" 
            produces "output123.csv" from "video_123.mp4" in the above 
            in_pattern). Note that file extension is optional and will always
            be replaced with the appropriate extensions. If neither None or
            a string, out_pattern should be sequence of string, and 
            substitution will be made separately for csv, figure, and video, 
            using the substitution patterns in that order.
          summary_name: name of the summary file. Not used unless the 
            "summary" option is true in the "output" section of config
          stdout: function to write verbose messages to
        
        RETURNS None
        
        SIDE EFFECTS:
          files output to out_folder
        '''
        
        # check if the in_folder and out_folder are actual directories
        if not pathlib.Path(in_folder).is_dir():
            raise FileNotFoundError("'{}' is not a folder".format(in_folder))
        if not pathlib.Path(out_folder).is_dir():
            raise FileNotFoundError("'{}' is not a folder".format(out_folder))
        
        if self.config["modes"]["verbosity"] > 1:
            stdout("Setting up main loop...")
        
        # determine mode to interpret __call__() input
        if self.config["modes"]["re"]:
            generator = self._re_generator(
                in_folder, out_folder, in_pattern, out_pattern,
                self.config["output"]["figure_format"]
            )
        else:
            generator = self._glob_generator(
                in_folder, out_folder, in_pattern,
                self.config["output"]["figure_format"]
            )
        
        # determine the full path of summary file
        smry_path = pathlib.Path(summary_name).with_suffix(".csv")
        if not smry_path.is_absolute():
            smry_path = pathlib.Path(out_folder) / smry_path
        
        # call internal _run() method for the real work
        self._run(generator, smry_path, stdout=stdout)
        
        if self.config["modes"]["verbosity"]:
            stdout("ALL DONE!")
        
        return

# control the import * behavior
__all__ = ["SingleObjBatch", "MultiObjBatch"]

# allow the batch module to be run from command line
if __name__=="__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch Video Motion Processing'
    )
    parser.add_argument('in_folder', help='input folder for source videos')
    parser.add_argument('out_folder', help='output folder for results')
    parser.add_argument('-m', '--mode', default="single", help=(
        'object detection mode ("single" or "multi"); ' + 
        'default="single"'
    ))
    parser.add_argument('-c', '--config', default=None, 
        help='configuration .json file')
    parser.add_argument('-i', '--in_pattern', default=None, 
        help='pattern to filter input files')
    parser.add_argument('-o', '--out_pattern', default=None, 
        help='pattern to rename output files')
    parser.add_argument('-s', '--summary', default="summary.csv", 
        help='name of summary file')
    
    args = parser.parse_args()
    
    mode = args.mode.lower()
    if mode=="single":
        processor = SingleObjBatch(args.config)
    elif mode=="multi":
        processor = MultiObjBatch(args.config)
    else:
        raise ValueError("unknown object detection mode {}".format(args.mode))
    
    # convert postprocessing specification from config into actual pipeline
    processor.infer_post_ops(clear=True)
    
    # start postprocessing
    processor(
        args.in_folder, args.out_folder, args.in_pattern, args.out_pattern,
        args.summary
    )


