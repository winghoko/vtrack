'''
module to support graphical user interface 

The main purpose of this module is to provide Tkinter based graphical user
interface (GUI) for video motion tracking. There are 4 main GUIs defined in
this module, each implemented as a class. Two are for single-object tracking:
SingleObjOneGUI (for processing a single file) SingleObjBatchGUI (for 
processing multiple files). The remaining two classes are the multi-object 
tracking counterparts: MutliObjOneGUI (single file) and MultiObjBatchGUI 
(batch).

In addition, this module provide a GUI for choosing file/directory (the 
file_chooser() function) as well as various utilities to manage the global
Tk instance that enables these GUIs.

NOTE: when this module is first imported it initialize a top-level Tk instance
that can be accessed by the .root attribute of the module. In general, 
functions in this module may access and manipulate this instance as a 
global variable.

COMMAND LINE USE: the GUIs defined in the module can be run directly
from command line via (e.g.,) `python -m vtrack.gui [options]`. This basically 
instantize the appropriate GUI class and then calls it with the supplied 
command line inputs. For the mapping of command line inputs to underlying 
function arguments, use the -h (or --help) option in command line
'''

# import modules from python STL
import pathlib, abc
import tkinter, tkinter.filedialog, tkinter.scrolledtext, tkinter.messagebox
from json.decoder import JSONDecodeError
from ast import literal_eval

# try to clean up previous instance ( useful for importlib.reload() ) 
try:
    root
except NameError:
    root = None

if root is not None:
    try:
        root.destroy()
    except tkinter.TclError:
        pass

# initialize a module-level Tk root and immediately hide it
root = tkinter.Tk()
root.withdraw()
root.update()

#### Convenient functions to manage Tk root

def unhide():
    '''
    Unhide the module-global top-level Tk instance
    
    NO ARGUMENTS and NO RETURNS
    
    SIDE EFFECTS:
      the module-global top-level Tk instance is hidden if exists
    '''
    global root
    if root is not None:
        root.deiconify()

def hide():
    '''
    Hide the module-global top-level Tk instance
    
    NO ARGUMENTS and NO RETURNS
    
    SIDE EFFECTS:
      the module-global top-level Tk instance is hidden if exists
    '''
    global root
    if root is not None:
        root.withdraw()

def terminate():
    '''
    Destroy (terminate) the module-global top-level Tk instance
    
    NO ARGUMENTS and NO RETURNS
    
    SIDE EFFECTS:
      the module-global top-level Tk instance is destroyed if exists;
      the module-global variable root is assigned to None
    '''
    global root

    if root is not None:
        try:
            root.destroy()
        except tkinter.TclError:
            pass

        root = None

    return

def reinitialize():
    '''
    Destroy (terminate) the module-global top-level Tk instance if exists, 
    then create a new module-global top-level Tk instance and assign it to root
    
    NO ARGUMENTS and NO RETURNS
    
    SIDE EFFECTS:
      the module-global top-level Tk instance is destroyed if exists;
      a new module-global top-level Tk instance is created and assigned to the
      variable root
    '''
    global root

    if root is not None:
        try:
            root.destroy()
        except tkinter.TclError:
            pass

    root = tkinter.Tk()
    root.withdraw()
    root.update()

    return

#### Standalone GUI functions

def file_chooser(mode, *, raise_error=True, **kwargs):
    '''
    file chooser that opens the OS's GUI dialog for choosing a file/directory
    
    ARGUMENTS: 
      mode: "r" for file to read, "w" for file to save, "d" for directory
      raise_error: whether to raise error if no files/directories are chosen
      (**kwargs): remaining keyword arguments are passed to the underlying
        tkinter.filedialog function
    
    RETUNRS: 
      pathlib.Path object (or tuple of pathlib.Path objects if multiple==True
      in **kwargs) of absolute file or directory paths
    
    NOTES:
      this function is essentially a convenient wrapper packing several
      tkinter.filedialog functions into one
    '''

    # reinitialize Tk root if necessary
    if root is None:
        reinitialize()

    # route to different tkinter.filedialog functions depending on mode
    if mode.lower()=="r":
        out = tkinter.filedialog.askopenfilename(**kwargs)
    elif mode.lower()=="w":
        out = tkinter.filedialog.asksaveasfilename(**kwargs)
    elif mode.lower()=="d":
        out = tkinter.filedialog.askdirectory(**kwargs)

    # empty string or empty tuple
    if raise_error and (not out):
        raise ValueError("selection is empty")

    # convert to pathlib.Path objects
    if type(out)==str:
        out = pathlib.Path(out)
    else:
        out = tuple([ pathlib.Path(__) for __ in out ])

    return out

#### Convenient functions to be passed as arguments to other functions

def NoOps(*args, **kwargs):
    '''
    Convenient function that does nothing and return None
    '''
    return

def IdenOp(arg):
    '''
    Convenient function for identity operation
    '''
    return arg

def trim(docstring):
    '''
    trim a multi-line string in the manner in which docstrings are trimmed
    Source: [PEP 257](https://www.python.org/dev/peps/pep-0257/), from the
    subsection titled "Handling Docstring Indentation"
    '''
    if not docstring:
        return ''
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    maxint = len(docstring)
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = maxint
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < maxint:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)
    # Return a single string:
    return '\n'.join(trimmed)

#### Function factories for tkinter widgets

def make_filedialog(mode, callback=NoOps, *, raise_error=False, **kwargs):
    '''
    function factory for popping up file dialog and invoke callback on return;
    intended to be used as a factory of tkinter callbacks
    
    ARGUMENTS:
      mode: "r" for file to read, "w" for file to save, "d" for directory
      callback: the callback function to execute on the returned string 
        (or tuple of strings)
      raise_error: whether to raise error if no files/directories are chosen
      (**kwargs): remaining keyword arguments are passed to the underlying
        tkinter.filedialog function
    
    RETURNS:
      a function that opens a file dialog
    
    NOTE:
      assumes Tk root instance exists
    '''
    def filedialog():
        '''
        open a file dialog
        
        NO ARGUMENTS
          
        FROM ENCLOSING SCOPE:
          mode: "r" for file to read, "w" for file to save, "d" for directory
          callback: the callback function to execute on the returned string 
            (or tuple of strings)
          raise_error: whether to raise error if no files/directories are chosen
          (**kwargs): remaining keyword arguments are passed to the underlying
            tkinter.filedialog function
        '''
        # route to different tkinter.filedialog functions depending on mode
        mode_l = mode.lower()
        if mode_l=="r":
            out = tkinter.filedialog.askopenfilename(**kwargs)
        elif mode_l=="w":
            out = tkinter.filedialog.asksaveasfilename(**kwargs)
        elif mode_l=="d":
            out = tkinter.filedialog.askdirectory(**kwargs)

        # empty string or empty tuple
        if raise_error and (not out):
            raise ValueError("selection is empty")

        # assign result
        callback(out)

    return filedialog

def make_elem_insert(elem, position, *, clear=True):
    '''
    function factory for inserting a string into an element at given position;
    intended to be used as a factory of tkinter callbacks
    
    ARGUMENTS:
      elem: tkinter element to be inserted
      position: position to insert string
      clear: whether to clear existing texts in the element before insertion
    
    RETURNS:
      a function taking a string as input and insert it into element 
      at position
    '''
    def elem_insert(string):
        '''
        insert string into the tkinter element at specified position
        
        ARGUMENTS:
          string: string to be inserted
          
        FROM ENCLOSING SCOPE:
          elem: tkinter element to be inserted
          position: position to insert string
          clear: whether to clear existing texts in element before insertion
        '''
        if clear:
            elem.delete(0, tkinter.END)

        elem.insert(position, string)

    return elem_insert

def make_message_pop(
    master, width, height, data, *, 
    geometry=None, title=None, preprocess=None, **kwargs
):
    '''
    function factory for creating a standalone message box;
    intended to be used as a factory of tkinter callback
    
    ARGUMENTS:
      master: the parent of the message box windows
      width: width (in character unit) of the message box
      height: height (in character unit) of the message box
      data: position to insert string
      geometry: the geometry of the message box
      preprocess: pre-processing to be applied to data (None if data is 
        used as-is)
      (**kwargs): arguments pass to the ScrollText constructor
    
    RETURNS:
      a function taking no input and create a pop-up message box
    
    NOTE:
      assumes Tk root instance exists
    '''

    if preprocess is not None:
        data = preprocess(data)

    def message_pop():
        '''
        insert string into the tkinter element at specified position
        
        ARGUMENTS:
          string: string to be inserted
        
        FROM ENCLOSING SCOPE:
          master: the parent of the message box windows
          width: width (in character unit) of the message box
          height: height (in character unit) of the message box
          data: position to insert string
          geometry: the geometry of the message box
          preprocess: pre-processing to be applied to data (None if data is 
            used as-is)
          (**kwargs): arguments pass to the ScrollText constructor
        '''
        window = tkinter.Toplevel(master=master)
        if title is not None:
            window.title(title)
        if geometry is not None:
            window.geometry(geometry)
        textbox = tkinter.scrolledtext.ScrolledText(
            master=window, relief="ridge", width=width, height=height, 
            **kwargs
        )
        textbox.insert(tkinter.END, data)
        textbox.configure(state='disabled')
        textbox.pack(fill=tkinter.BOTH, expand=True)
        window.update()

    return message_pop

#### Classes of full-fledged GUIs

class BaseBatchGUI(abc.ABC):
    '''
    Abstract base class for batch processing GUIs
    NOTE: has CONCRETE methods nonetheless!
    '''

    # help for get_option()
    get_opt_help = '''
    To see all currently defined options, leave "section" empty and hit "Get!"
    
    To see all options defined within a section and their values, enter the
    section in "section" and leave "option" empty, then hit "Get!"
    
    To see the value of a particular option, enter both section and option
    in their respective entry boxes and hit "Get!"
    
    To close the Get Option(s) GUI window, hit "Back"
    '''
    
    # help for set_option()
    set_opt_help = '''
    To set the value of a particular option, enter both section and option
    in their respective entry boxes, and enter the desired value into the
    "value" entry box, then hit "Set!".
    
    The option to be set need not be in the current configuration: a new option 
    will be created if none currently existed (note however that the backend
    may ignore irrelevant options).
    
    The input to the "section" and "option" entry boxes will be treated as
    python string. In contrast, the input to the "value" entry box will be 
    interpreted as a python literal expression if possible, and interpreted
    as string only if such attempt fails.
    
    In practice, this means that floating point number should include decimal 
    place, and string can optionally be surrounded by a pair of (single or 
    double) quotes for disambiguation.
    
    To close the Set Option(s) GUI window, hit "Back"
    '''

    def load_config_callback(self, config_file):
        '''
        to be called by filedialog (constructed by make_filedialog) to 
        update the actual in-memory configuration at the backend
        '''

        try:
            self.backend.load_config(config_file)

        # show info to user using appropriate message boxes
        except FileNotFoundError:
            tkinter.messagebox.showerror(
            title="load_config", 
            message="File not found. Configuration not loaded"
            )
        except JSONDecodeError:
            tkinter.messagebox.showerror(
            title="load_config", 
            message="Invalid .json file. Configuration not loaded"
            )
        else:
            tkinter.messagebox.showinfo(
                title="load_config", 
                message="Configuration file loaded"
            )

    def save_config_callback(self, config_file):
        '''
        to be called by filedialog (constructed by make_filedialog) to 
        actually output the in-memory configuration at the backend to file
        '''

        try:
            self.backend.save_config(config_file, indent=2)

        # show info to user using appropriate message boxes
        except FileNotFoundError:
            tkinter.messagebox.showerror(
            title="load_config", 
            message="File not found. Configuration not saved"
            )
        else:
            tkinter.messagebox.showinfo(
                title="save_config", 
                message="Configuration file saved"
            )

    def reset_config(self):
        '''
        reset the in-memory configuration at the backend to the class-level
        default
        '''

        self.backend.copy_defaults()

        # show info to user using appropriate message boxes
        tkinter.messagebox.showinfo(
            title="reset_config", 
            message="Default configuration restored"
        )

    def start_get_GUI(self):
        '''
        launch the GUI for getting config option values to a new window
        '''
        # start new top-level window
        get_GUI = tkinter.Toplevel(master=self.frontend)
        get_GUI.title("Get Option(s) GUI")

        # two organizing frames
        frm_get_opt = tkinter.Frame(master=get_GUI)
        frm_get_prompt = tkinter.Frame(master=get_GUI)

        # grid structure of top frame
        frm_get_opt.columnconfigure(0, weight=0, minsize=50)
        frm_get_opt.columnconfigure(1, weight=1, minsize=250)

        # left of top frame: labels
        lbl_get_sect = tkinter.Label(
            master=frm_get_opt, text="Section:", width=10
        )
        lbl_get_sect.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        lbl_get_opt = tkinter.Label(
            master=frm_get_opt, text="Option:", width=10
        )
        lbl_get_opt.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        # right of top frame: entry boxes
        entr_get_sect = tkinter.Entry(master=frm_get_opt)
        entr_get_sect.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        entr_get_opt = tkinter.Entry(master=frm_get_opt)
        entr_get_opt.grid(row=1, column=1, padx=5, pady=5, sticky="we")

        # configure the geometry of the second frame
        frm_get_prompt.columnconfigure(0, weight=0)
        frm_get_prompt.columnconfigure(1, weight=0)
        frm_get_prompt.columnconfigure(2, weight=1)

        # second frame: three buttons
        btn_get = tkinter.Button(
            master=frm_get_prompt, text="Get!", width=5,
            command=self.get_option
        )
        btn_get.grid(row=0, column=0, padx=10, pady=5)
        btn_get_exit = tkinter.Button(
            master=frm_get_prompt, text="Back", width=5,
            command=self.kill_get_GUI
        )
        btn_get_exit.grid(row=0, column=1, padx=10, pady=5)
        btn_get_help = tkinter.Button(
            master=frm_get_prompt, text="Help", width=5,
            command=make_message_pop(
                self.frontend, 80, 15, self.get_opt_help, 
                title="Help: Get Option", geometry="640x240", 
                preprocess=trim, padx=5, pady=5, bg="white"
            )
        )
        btn_get_help.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        # implicit third organizing frame: output textbox
        get_outbox = tkinter.scrolledtext.ScrolledText(
            master=get_GUI, relief="ridge", width=80, height=5, 
            padx=5, pady=5, bg="white"
        )
        get_outbox.configure(state='disabled')

        # pack the organizing frames to the window
        frm_get_opt.pack(fill=tkinter.BOTH, expand=False)
        frm_get_prompt.pack(fill=tkinter.BOTH, expand=False)
        get_outbox.pack(fill=tkinter.BOTH, expand=True)

        # define protocol when user closed the GUI window
        get_GUI.protocol("WM_DELETE_WINDOW", self.kill_get_GUI)

        # "launch" the GUI
        get_GUI.geometry("480x240")
        get_GUI.update()

        # hook local variable to instance attribute
        self.get_GUI = get_GUI
        self.get_entries = {
            "section": entr_get_sect,
            "option": entr_get_opt
        }
        self.get_outbox = get_outbox

    def get_option(self):
        '''
        obtain the request for getting config option from the GUI, 
        query the backend, and display the result in the GUI
        '''
        # obtain (string) arguments from the respective entry boxes
        args = [ __.get().strip() for __ in self.get_entries.values() ]

        if not args[0]: #empty or None
            args[0] = None
        if not args[1]: #empty or None
            args[1] = None

        try:
            out = self.backend.get_option(*args)
        except KeyError:
            out = "[WARNING] value for ({}, {}) not found!".format(
                *args
            )

        if type(out)==dict: # format output if a whole section is requested
            out = "\n".join(
                key + ": " + str(value) for (key, value) in out.items()
            )
        elif type(out) != str:
            out = str(out)

        # output to textbox
        self.get_outbox.configure(state='normal')
        self.get_outbox.delete("0.0", tkinter.END)
        self.get_outbox.insert(tkinter.END, out)
        self.get_outbox.configure(state='disabled')
        self.get_GUI.update()

    def kill_get_GUI(self):
        '''
        terminate the window containing the GUI for getting the values of
        config options
        '''
        self.get_outbox = None
        self.get_entries = {
            "section": None,
            "option": None
        }
        if self.get_GUI is not None:
            self.get_GUI.destroy()
            self.get_GUI = None

    def start_set_GUI(self):
        '''
        launch the GUI for setting config option values to a new window
        '''
        # start new top-level window
        set_GUI = tkinter.Toplevel(master=self.frontend)
        set_GUI.title("Set Option(s) GUI")

        # two organizing frames
        frm_set_opt = tkinter.Frame(master=set_GUI)
        frm_set_prompt = tkinter.Frame(master=set_GUI)

        # grid structure of top frame
        frm_set_opt.columnconfigure(0, weight=0, minsize=50)
        frm_set_opt.columnconfigure(1, weight=1, minsize=250)

        # left of top frame: labels
        lbl_set_sect = tkinter.Label(
            master=frm_set_opt, text="Section:", width=10
        )
        lbl_set_sect.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        lbl_set_opt = tkinter.Label(
            master=frm_set_opt, text="Option:", width=10
        )
        lbl_set_opt.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        lbl_set_val = tkinter.Label(
            master=frm_set_opt, text="Value:", width=10
        )
        lbl_set_val.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        # right of top frame: entry boxes
        entr_set_sect = tkinter.Entry(master=frm_set_opt)
        entr_set_sect.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        entr_set_opt = tkinter.Entry(master=frm_set_opt)
        entr_set_opt.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        entr_set_val = tkinter.Entry(master=frm_set_opt)
        entr_set_val.grid(row=2, column=1, padx=5, pady=5, sticky="we")

        # configure the geometry of the second frame
        frm_set_prompt.columnconfigure(0, weight=0)
        frm_set_prompt.columnconfigure(1, weight=0)
        frm_set_prompt.columnconfigure(2, weight=1)

        # second frame: three buttons
        btn_set = tkinter.Button(
            master=frm_set_prompt, text="Set!", width=5,
            command=self.set_option
        )
        btn_set.grid(row=0, column=0, padx=10, pady=5)
        btn_set_exit = tkinter.Button(
            master=frm_set_prompt, text="Back", width=5,
            command=self.kill_set_GUI
        )
        btn_set_exit.grid(row=0, column=1, padx=10, pady=5)
        btn_set_help = tkinter.Button(
            master=frm_set_prompt, text="Help", width=5,
            command=make_message_pop(
                self.frontend, 80, 15, self.set_opt_help, 
                title="Help: Set Option", geometry="640x360", 
                preprocess=trim, padx=5, pady=5, bg="white"
            )
        )
        btn_set_help.grid(row=0, column=2, padx=5, pady=5, sticky="e")

        # pack the organizing frames to the window
        frm_set_opt.pack(fill=tkinter.BOTH, expand=False)
        frm_set_prompt.pack(fill=tkinter.BOTH, expand=False)

        # define protocol when user closed the GUI window
        set_GUI.protocol("WM_DELETE_WINDOW", self.kill_set_GUI)

        # "launch" the GUI
        set_GUI.geometry("480x140")
        set_GUI.update()

        # hook local variable to instance attributes
        self.set_GUI = set_GUI
        self.set_entries = {
            "section": entr_set_sect,
            "option": entr_set_opt,
            "value": entr_set_val
        }

    def set_option(self):
        '''
        obtain the request for setting config option from the GUI, 
        and route the request to the backend
        '''

        # obtain (string) arguments from the respective entry boxes
        args = [ __.get() for __ in self.set_entries.values() ]

        args[0] = args[0].strip()
        args[1] = args[1].strip()

        if (not args[0]) or (not args[1]):
            tkinter.messagebox.showerror(
                title="set_option", 
                message="Invalid option specification. No action taken"
            )
            return

        val = args[2]
        try:
            args[2] = literal_eval(val)
        except Exception as e: # restore default type of string
            args[2] = val

        self.backend.set_option(*args)

        # show confirmation on option setting via message box
        tkinter.messagebox.showinfo(
            title="set_option", 
            message="Value {2} saved for ({0}, {1})".format(*args)
        )

    def kill_set_GUI(self):
        '''
        terminate the window containing the GUI for setting values of 
        config options
        '''
        self.set_entries = {
            "section": None,
            "option": None,
            "value": None
        }
        if self.set_GUI is not None:
            self.set_GUI.destroy()
            self.set_GUI = None

    @abc.abstractmethod
    def start_frontend(self):
        '''
        start the main frontend GUI and the Tk mainloop
        '''
        pass

    @abc.abstractmethod
    def kill_frontend(self):
        '''
        terminate the main GUI (and all of its children) and restart the root
        Tk window
        '''
        pass

    @abc.abstractmethod
    def stdout(self, data):
        '''
        output the given data to the "standard output" of the main GUI. To
        be used by the backend to route output
        '''
        pass

    @abc.abstractmethod
    def process(self):
        '''
        route the request to start new video motion tracking to the backend
        '''
        pass

class SingleObjOneGUI(BaseBatchGUI):
    '''
    class to implement graphical user interface (GUI) for single object,
    single file video motion tracking
    '''

    # help message to be displayed
    main_help = '''
    Graphical user interface (GUI) for single object, single file video 
    motion tracking
    
    To run, select the input file and output files (either through the file 
    dialog accessible via the associated button, or by directly entering the 
    file path into the text box), and then click "RUN!" to process.
    
    NOTE that an omitted output filename instructs the GUI to omit the 
    corresponding file output.
    
    The detailed setting of the video processing pipeline is determined by
    the configuration of the internal batch processing backend. The 
    configuration can be adjusted as a whole by:
      1/ loading a new configuration ("Load Config") from a .json file
      2/ reseting to an internal default ("Reset Config")
    Furthermore, the current configuration can be saved as a .json file for 
    future use via "Save Config".
    
    The configuration (henceforth "config" in short) can be considered as
    consisting of multiple "sections", each contain a number of "options".
    The current config can be viewed at a per-option basis using "Get Option", 
    and can be set at a per-option basis using "Set Option".
    
    The config consists of 4 mandatory sections termed "modes", "output", 
    "params", and "template_texts". As a general rule, all options within these
    4 sections should be supplied in the config file, EVEN IF the option is
    NOT being used (e.g., "figure_format" in "output" should be set even if 
    there is no figure to be outputted).
    
    The options under the "modes" section are:
      verbosity: (0, 1, or 2) how verbose is the messages printed to stdout 
        while the batch processing happens
      re: (bool) whether the patterns supplied when "RUN!" is clicked are 
        interpreted as regular expression or simple wildcard-enabled string
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
    
    There are 8 options under the "output" section, 5 of which are 
    AUTOMATICALLY set by the user input of filenames in the main GUI when
    "RUN!" is executed:
      details: (bool) whether the detailed motion data is output to csv.
        True if GUI input of "CSV Output" is non-empty; False otherwise 
      summary: (bool) whether the summary of motion data (one line for each
        file) is output to csv. True if GUI input of "Summary Output" is 
        non-empty; False otherwise
      figure: (bool) whether to output (save as file) the main plot of the 
        path taken by the object. True if GUI input of "Figure Output" is 
        non-empty; False otherwise
      video: (bool) whether to generate an overlaid video. True if GUI input 
        of "Video Output" is non-empty; False otherwise
      figure_format: the format (file extension) for which the main plot will
        be saved in. Follow the file extension in the GUI input of "Figure 
        Output"
    The remaining 3 options under the "output" section are:
      record_threshold: (bool) if True, the threshold value will be saved to 
        the summary file
      record_conversion: (bool) if True, the pixel-to-physical unit conversion 
        factor will be asved to the summary file
      record_boxes: (bool) if True, the cropbox, maskbox, and innerbox 
        coordinates will be saved to the summary file
    
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
      automatically determined as the smallest rectangle enclosing the mask.
    
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
    "plot_path", and "export_overlaid_video".
    
    Finally, the config may contain a "postprocess" section specifying 
    standard post-processing to be performed after binary frames are obtained.
    The section has 2 options:
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
    '''

    def __init__(self, config_file=None):
        '''
        initialize the main GUI after asking for the configuration file
        '''

        from .batch import SingleObjBatch

        # initialize GUI
        if root is None:
            reinitialize()

        # choose configuration file
        if config_file is None:
            try:
                config = file_chooser(
                    "r", title="Select configuration file..."
                )
            except ValueError as e:
                config = None
        else:
            config = config_file

        # initialize batch processor
        try:
            self.backend = SingleObjBatch(config)

        # show info to user using appropriate message boxes
        except JSONDecodeError as e:
            tkinter.messagebox.showerror(
                title="load_config", 
                message="Invalid .json file. Configuration not loaded"
            )
        else:
            if config is None:
                tkinter.messagebox.showinfo(
                    title="load_config", 
                    message="No file supplied. Default configuration used"
                )
            else:
                tkinter.messagebox.showinfo(
                    title="load_config", 
                    message="Configuration file loaded"
                )

        # initialize GUI
        self.start_frontend()

    def figfile_dialog(self):
        '''
        start a file dialog in which the default extension is obtained live
        from the current backend configuration
        '''
        try:
            # get current output extension
            ext = self.backend.get_option("output", "figure_format")
        except KeyError:
            # file dialog without default extension
            out = tkinter.filedialog.asksaveasfilename()
        else:
            # file dialog with default extension
            out = tkinter.filedialog.asksaveasfilename(
                defaultextension=ext
            )

        # assign result
        self.entries["fig_path"].delete(0, tkinter.END)
        self.entries["fig_path"].insert(tkinter.END, out)

    def start_frontend(self):

        # reinitialize Tk root if none existed
        if root is None:
            reinitialize()

        # create window for main GUI
        frontend = tkinter.Toplevel(master=root)
        frontend.title("Single Object Per-File Processing GUI")

        # creating three organizing frames
        frm_config = tkinter.Frame(master=frontend)
        frm_files = tkinter.Frame(master=frontend)
        frm_prompt = tkinter.Frame(master=frontend)

        # configure the grid of the first frame
        frm_config.columnconfigure(0, weight=0)
        frm_config.columnconfigure(1, weight=0)
        frm_config.columnconfigure(2, weight=0)
        frm_config.columnconfigure(3, weight=1)

        # first frame: two rows of buttons
        btn_load_conf = tkinter.Button(
            master=frm_config, text="Load Config", width=12,
            command=make_filedialog(
                "r", self.load_config_callback
            )
        )
        btn_load_conf.grid(row=0, column=0, padx=3, pady=3)
        btn_save_conf = tkinter.Button(
            master=frm_config, text="Save Config", width=12,
            command=make_filedialog(
                "w", self.save_config_callback, defaultextension=".json"
            )
        )
        btn_save_conf.grid(row=0, column=1, padx=3, pady=3)
        btn_reset_conf = tkinter.Button(
            master=frm_config, text="Reset Config", width=12,
            command=self.reset_config
        )
        btn_reset_conf.grid(row=0, column=2, padx=3, pady=3)
        btn_help = tkinter.Button(
            master=frm_config, text="Help", width=5,
            command=make_message_pop(
                frontend, 80, 15, self.main_help, title="Help", 
                geometry="640x480", preprocess=trim, padx=5, pady=5, 
                bg="white"
            )
        )
        btn_help.grid(row=0, column=3, padx=5, pady=3, sticky="e")
        btn_get_opt = tkinter.Button(
            master=frm_config, text="Get Option", width=12,
            command=self.start_get_GUI
        )
        btn_get_opt.grid(row=1, column=0, padx=3, pady=3)
        btn_set_opt = tkinter.Button(
            master=frm_config, text="Set Option", width=12,
            command=self.start_set_GUI
        )
        btn_set_opt.grid(row=1, column=1, padx=3, pady=3)

        # configure the grid of the second frame
        frm_files.columnconfigure(0, weight=0, minsize=150)
        frm_files.columnconfigure(1, weight=1, minsize=250)

        # right of second frame: entry boxes
        entr_input = tkinter.Entry(master=frm_files)
        entr_input.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        entr_csv = tkinter.Entry(master=frm_files)
        entr_csv.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        entr_fig = tkinter.Entry(master=frm_files)
        entr_fig.grid(row=2, column=1, padx=5, pady=5, sticky="we")
        entr_vid = tkinter.Entry(master=frm_files)
        entr_vid.grid(row=3, column=1, padx=5, pady=5, sticky="we")
        entr_smry = tkinter.Entry(master=frm_files)
        entr_smry.grid(row=4, column=1, padx=5, pady=5, sticky="we")

        # left of second frame: buttons
        btn_input = tkinter.Button(
            master=frm_files, text="Select Input File", width=20,
            command=make_filedialog(
                "r", make_elem_insert(
                    entr_input, tkinter.END, clear=True
                )
            )
        )
        btn_input.grid(row=0, column=0, padx=5, pady=5)
        btn_csv = tkinter.Button(
            master=frm_files, text="Select CSV Output", width=20,
            command=make_filedialog(
                "w", make_elem_insert(
                    entr_csv, tkinter.END, clear=True
                ), defaultextension=".csv"
            )
        )
        btn_csv.grid(row=1, column=0, padx=5, pady=5)
        btn_fig = tkinter.Button(
            master=frm_files, text="Select Figure Output", width=20,
            command=self.figfile_dialog
        )
        btn_fig.grid(row=2, column=0, padx=5, pady=5)
        btn_vid = tkinter.Button(
            master=frm_files, text="Select Video Output", width=20,
            command=make_filedialog(
                "w", make_elem_insert(
                    entr_vid, tkinter.END, clear=True
                ), defaultextension=".mp4"
            )
        )
        btn_vid.grid(row=3, column=0, padx=5, pady=5)
        btn_smry = tkinter.Button(
            master=frm_files, text="Select Summary Output", width=20,
            command=make_filedialog(
                "w", make_elem_insert(
                    entr_smry, tkinter.END, clear=True
                ), defaultextension=".csv"
            )
        )
        btn_smry.grid(row=4, column=0, padx=5, pady=5)

        # third frame: two buttons
        btn_run = tkinter.Button(
            master=frm_prompt, text="RUN!", width=5,
            command=self.process
        )
        btn_run.grid(row=0, column=0, padx=10, pady=5)
        btn_exit = tkinter.Button(
            master=frm_prompt, text="Exit", width=5,
            command=self.kill_frontend
        )
        btn_exit.grid(row=0, column=1, padx=10, pady=5)

        # implicit fourth frame: output text box
        outbox = tkinter.scrolledtext.ScrolledText(
            master=frontend, relief="ridge", width=80, height=5, 
            padx=5, pady=5, bg="white"
        )
        outbox.configure(state='disabled')

        # pack the organizing frames to the windows
        frm_config.pack(fill=tkinter.BOTH, expand=False)
        frm_files.pack(fill=tkinter.BOTH, expand=False)
        frm_prompt.pack(fill=tkinter.BOTH, expand=False)
        outbox.pack(fill=tkinter.BOTH, expand=True)

        # define protocol when user closed the GUI window
        frontend.protocol("WM_DELETE_WINDOW", self.kill_frontend)

        # "launch" the main GUI
        frontend.geometry("640x480")
        frontend.update()

        # make sure all sub-windows are initialized
        self.get_GUI = None
        self.set_GUI = None
        self.kill_get_GUI()
        self.kill_set_GUI()

        # hook local variable to instance attributes
        self.frontend = frontend
        self.outbox = outbox
        self.entries = {
            "in_path": entr_input,
            "csv_path": entr_csv,
            "fig_path": entr_fig,
            "vid_path": entr_vid,
            "smry_path": entr_smry
        }

        # start main loop (must be last line!)
        root.mainloop()

    def kill_frontend(self):

        # make sure all children windows are closed
        self.kill_get_GUI()
        self.kill_set_GUI()

        # reset instance attributes
        self.outbox = None
        self.entries = {
            "in_path": None,
            "csv_path": None,
            "fig_path": None,
            "vid_path": None,
            "smry_path": None
        }
        if self.frontend is not None:
            self.frontend.destroy()
            self.frontend = None

        # kill the current Tk mainloop
        terminate()

    def stdout(self, data, *, position=tkinter.END, end="\n"):

        self.outbox.configure(state='normal')
        self.outbox.insert(position, data + end)
        self.outbox.see(tkinter.END)
        self.outbox.configure(state='disabled')
        self.frontend.update()

    def infer_options(self):
        '''
        set config values based on user's GUI input
        '''

        # alias for easy modifications
        output = self.backend.config["output"]

        raw = [ __.get().strip() for __ in self.entries.values() ]

        if not raw[0]: # empty input file
            tkinter.messagebox.showerror(
                title="RUN!", 
                message="Empty input file. No action taken"
            )
            raise ValueError("Invalid input")

        if raw[1]:
            output["details"] = True
        else: # empty CSV file
            output["details"] = False

        if raw[2]:
            output["figure"] = True
            output["figure_format"] = pathlib.Path(raw[2]).suffix
        else: # empty figure file
            output["figure"] = False

        if raw[3]:
            output["video"] = True
        else: # empty video file
            output["video"] = False

        if raw[4]:
            output["summary"] = True
        else:
            output["summary"] = False

    def process(self):

        # infer config settings from user input
        try:
            self.infer_options()
        except ValueError:
            return

        paths = [ 
            pathlib.Path(__.get().strip()) for __ in self.entries.values() 
        ]

        # infer postprocessing from postprocess section of config
        self.backend.infer_post_ops(clear=True)

        # clear previous texts printed to outbox
        self.outbox.configure(state='normal')
        self.outbox.delete("0.0", tkinter.END)
        self.outbox.configure(state='disabled')

        # route to backend
        self.backend._run([ paths[:-1] ], paths[-1], stdout=self.stdout)

        # signal to user when all video processing is done
        self.stdout("DONE!")

class SingleObjBatchGUI(BaseBatchGUI):
    '''
    class to implement graphical user interface (GUI) for single object,
    multiple files video motion tracking
    '''

    # help message to be displayed
    main_help = '''
    Graphical user interface (GUI) for single object, multiple files video 
    motion tracking
    
    To run, select the input folder, output folder, and the path of the summary
    file (either through the file dialog accessible via the associated button, 
    or by directly entering the file path into the text box), as well as 
    pattern to match for input (e.g., "*.mp4" for all .mp4 file) and pattern 
    to substitute as output (NOTE: only meaningful in "re" mode; see below).
    Once these are selected, click "RUN!" to start batch processing.
    
    NOTE that while input and output folders are mandatory, the summary file 
    path can be omitted (in which case no summary file is saved). Similarly, 
    both in_pattern and out_pattern can be omitted. If in_pattern is omitted
    all files in in_folder are processed. If out_pattern is omitted in "re" 
    mode, the output files take the same name as the input file except possibly
    for the file extensions. The same convention for output filename also 
    applies in "non-re" mode regardless of out_pattern specification.
    
    The detailed setting of the video processing pipeline is determined by
    the configuration of the internal batch processing backend. The 
    configuration can be adjusted as a whole by:
      1/ loading a new configuration ("Load Config") from a .json file
      2/ reseting to an internal default ("Reset Config")
    Furthermore, the current configuration can be saved as a .json file for 
    future use via "Save Config".
    
    The configuration (henceforth "config" in short) can be considered as
    consisting of multiple "sections", each contain a number of "options".
    The current config can be viewed at a per-option basis using "Get Option", 
    and can be set at a per-option basis using "Set Option".
    
    The config consists of 4 mandatory sections termed "modes", "output", 
    "params", and "template_texts". As a general rule, all options within these
    4 sections should be supplied in the config file, EVEN IF the option is
    NOT being used (e.g., "figure_format" in "output" should be set even if 
    there is no figure to be outputted).
    
    The options under the "modes" section are:
      verbosity: (0, 1, or 2) how verbose is the messages printed to stdout 
        while the batch processing happens
      re: (bool) whether the patterns supplied when "RUN!" is clicked are 
        interpreted as regular expression or simple wildcard-enabled string
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
    
    There are 8 options under the "output" section. Of these, summary is 
    AUTOMATICALLY set by the user input on the summary file (in particular, 
    summary is omitted if the summary file is not set). The remaining 7 
    options are:
      details: (bool) whether the detailed motion data is output to csv
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
      automatically determined as the smallest rectangle enclosing the mask.
    
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
    "plot_path", and "export_overlaid_video".
    
    Finally, the config may contain a "postprocess" section specifying 
    standard post-processing to be performed after binary frames are obtained.
    The section has 2 options:
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
    '''

    def __init__(self, config_file=None):
        '''
        initialize the main GUI after asking for the configuration file
        '''

        from .batch import SingleObjBatch

        # initialize GUI
        if root is None:
            reinitialize()

        # choose configuration file
        if config_file is None:
            try:
                config = file_chooser(
                    "r", title="Select configuration file..."
                )
            except ValueError as e:
                config = None
        else:
            config = config_file

        # initialize batch processor
        try:
            self.backend = SingleObjBatch(config)

        # show info to user using appropriate message boxes
        except JSONDecodeError as e:
            tkinter.messagebox.showerror(
                title="load_config", 
                message="Invalid .json file. Configuration not loaded"
            )
        else:
            if config is None:
                tkinter.messagebox.showinfo(
                    title="load_config", 
                    message="No file supplied. Default configuration used"
                )
            else:
                tkinter.messagebox.showinfo(
                    title="load_config", 
                    message="Configuration file loaded"
                )

        # initialize GUI
        self.start_frontend()

    def start_frontend(self):

        # reinitialize Tk root if none existed
        if root is None:
            reinitialize()

        # create window for main GUI
        frontend = tkinter.Toplevel(master=root)
        frontend.title("Single Object Batch Processing GUI")

        # creating three organizing frames
        frm_config = tkinter.Frame(master=frontend)
        frm_files = tkinter.Frame(master=frontend)
        frm_prompt = tkinter.Frame(master=frontend)

        # configure the grid of the first frame
        frm_config.columnconfigure(0, weight=0)
        frm_config.columnconfigure(1, weight=0)
        frm_config.columnconfigure(2, weight=0)
        frm_config.columnconfigure(3, weight=1)

        # first frame: two rows of buttons
        btn_load_conf = tkinter.Button(
            master=frm_config, text="Load Config", width=12,
            command=make_filedialog(
                "r", self.load_config_callback
            )
        )
        btn_load_conf.grid(row=0, column=0, padx=3, pady=3)
        btn_save_conf = tkinter.Button(
            master=frm_config, text="Save Config", width=12,
            command=make_filedialog(
                "w", self.save_config_callback, defaultextension=".json"
            )
        )
        btn_save_conf.grid(row=0, column=1, padx=3, pady=3)
        btn_reset_conf = tkinter.Button(
            master=frm_config, text="Reset Config", width=12,
            command=self.reset_config
        )
        btn_reset_conf.grid(row=0, column=2, padx=3, pady=3)
        btn_help = tkinter.Button(
            master=frm_config, text="Help", width=5,
            command=make_message_pop(
                frontend, 80, 15, self.main_help, title="Help", 
                geometry="640x480", preprocess=trim, padx=5, pady=5, 
                bg="white"
            )
        )
        btn_help.grid(row=0, column=3, padx=5, pady=3, sticky="e")
        btn_get_opt = tkinter.Button(
            master=frm_config, text="Get Option", width=12,
            command=self.start_get_GUI
        )
        btn_get_opt.grid(row=1, column=0, padx=3, pady=3)
        btn_set_opt = tkinter.Button(
            master=frm_config, text="Set Option", width=12,
            command=self.start_set_GUI
        )
        btn_set_opt.grid(row=1, column=1, padx=3, pady=3)

        # configure the grid of the second frame
        frm_files.columnconfigure(0, weight=0, minsize=150)
        frm_files.columnconfigure(1, weight=1, minsize=250)

        # right of second frame (top): entry boxes
        entr_infolder = tkinter.Entry(master=frm_files)
        entr_infolder.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        entr_outfolder = tkinter.Entry(master=frm_files)
        entr_outfolder.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        entr_summary = tkinter.Entry(master=frm_files)
        entr_summary.grid(row=2, column=1, padx=5, pady=5, sticky="we")

        # left of second frame (top): buttons
        btn_infolder = tkinter.Button(
            master=frm_files, text="Select Input Folder", width=20,
            command=make_filedialog(
                "d", make_elem_insert(
                    entr_infolder, tkinter.END, clear=True
                ), mustexist=True
            )
        )
        btn_infolder.grid(row=0, column=0, padx=5, pady=5)
        btn_outfolder = tkinter.Button(
            master=frm_files, text="Select Output Folder", width=20,
            command=make_filedialog(
                "d", make_elem_insert(
                    entr_outfolder, tkinter.END, clear=True
                ), mustexist=True
            )
        )
        btn_outfolder.grid(row=1, column=0, padx=5, pady=5)
        btn_summary = tkinter.Button(
            master=frm_files, text="Select Summary File", width=20,
            command=make_filedialog(
                "w", make_elem_insert(
                    entr_summary, tkinter.END, clear=True
                ), defaultextension=".csv"
            )
        )
        btn_summary.grid(row=2, column=0, padx=5, pady=5)

        # left of second frame (bottom): buttons
        lbl_inpattern = tkinter.Label(
            master=frm_files, text="Input Pattern:", width=20
        )
        lbl_inpattern.grid(row=3, column=0, padx=5, pady=5)
        lbl_outpattern = tkinter.Label(
            master=frm_files, text="Output Pattern:", width=20
        )
        lbl_outpattern.grid(row=4, column=0, padx=5, pady=5)

        # right of second frame (bottom): entry boxes
        entr_inpattern = tkinter.Entry(master=frm_files)
        entr_inpattern.grid(row=3, column=1, padx=5, pady=5, sticky="we")
        entr_outpattern = tkinter.Entry(master=frm_files)
        entr_outpattern.grid(row=4, column=1, padx=5, pady=5, sticky="we")

        # third frame: two buttons
        btn_run = tkinter.Button(
            master=frm_prompt, text="RUN!", width=5,
            command=self.process
        )
        btn_run.grid(row=0, column=0, padx=10, pady=5)
        btn_exit = tkinter.Button(
            master=frm_prompt, text="Exit", width=5,
            command=self.kill_frontend
        )
        btn_exit.grid(row=0, column=1, padx=10, pady=5)

        # implicit fourth frame: output text box
        outbox = tkinter.scrolledtext.ScrolledText(
            master=frontend, relief="ridge", width=80, height=5, 
            padx=5, pady=5, bg="white"
        )
        outbox.configure(state='disabled')

        # pack the organizing frames to the windows
        frm_config.pack(fill=tkinter.BOTH, expand=False)
        frm_files.pack(fill=tkinter.BOTH, expand=False)
        frm_prompt.pack(fill=tkinter.BOTH, expand=False)
        outbox.pack(fill=tkinter.BOTH, expand=True)

        # define protocol when user closed the GUI window
        frontend.protocol("WM_DELETE_WINDOW", self.kill_frontend)

        # "launch" the main GUI
        frontend.geometry("640x480")
        frontend.update()

        # make sure all sub-windows are initialized
        self.get_GUI = None
        self.set_GUI = None
        self.kill_get_GUI()
        self.kill_set_GUI()

        # hook local variable to instance attributes
        self.frontend = frontend
        self.outbox = outbox
        self.entries = {
            "in_folder": entr_infolder,
            "out_folder": entr_outfolder,
            "in_pattern": entr_inpattern,
            "out_pattern": entr_outpattern,
            "summary_name": entr_summary
        }

        # start main loop (must be last line!)
        root.mainloop()

    def kill_frontend(self):

        # make sure all children windows are closed
        self.kill_get_GUI()
        self.kill_set_GUI()

        # reset instance attributes
        self.outbox = None
        self.entries = {
            "in_folder": None,
            "out_folder": None,
            "in_pattern": None,
            "out_pattern": None,
            "summary_name": None
        }
        if self.frontend is not None:
            self.frontend.destroy()
            self.frontend = None

        # kill the current Tk mainloop
        terminate()

    def stdout(self, data, *, position=tkinter.END, end="\n"):

        self.outbox.configure(state='normal')
        self.outbox.insert(position, data + end)
        self.outbox.see(tkinter.END)
        self.outbox.configure(state='disabled')
        self.frontend.update()

    def infer_options(self):
        '''
        set config values based on user's GUI input
        '''

        # infer the value of summary in the "output" section
        raw = self.entries["summary_name"].get().strip()
        if raw:
            self.backend.config["output"]["summary"] = True
        else:
            self.backend.config["output"]["summary"] = False

    def process(self):

        # infer config settings from user input
        try:
            self.infer_options()
        except ValueError:
            return

        # obtain (string) arguments from the GUI
        args = [ __.get().strip() for __ in self.entries.values() ]
        args = args[:5]

        # check if input/output folders are specified
        if (not args[0]) or (not args[1]):
            tkinter.messagebox.showerror(
                title="RUN!", 
                message=(
                    "Both Input and Output folders must be specified. " +
                    "No action taken"
                )
            )
            return

        # convert empty strings to default values
        if not args[2]: args[2] = None
        if not args[3]: args[3] = None
        if not args[4]: args = args[:4]

        # infer postprocessing from postprocess section of config
        self.backend.infer_post_ops(clear=True)

        # clear previous texts printed to outbox
        self.outbox.configure(state='normal')
        self.outbox.delete("0.0", tkinter.END)
        self.outbox.configure(state='disabled')

        # route operations to the backend
        self.backend(*args, stdout=self.stdout)

class MultiObjOneGUI(BaseBatchGUI):
    '''
    class to implement graphical user interface (GUI) for multiple objects,
    single file video motion tracking
    '''

    # help message to be displayed
    main_help = '''
    Graphical user interface (GUI) for multiple objects, single file video 
    motion tracking
    
    To run, select the input file and the output files (either through the 
    file dialog accessible via the associated button, or by directly
    entering the file path into the text box), as well as the number of 
    foreground objects to be tracked, then click "RUN!" to process.
    
    NOTE that an omitted output filename instructs the GUI to omit the 
    corresponding file output.
    
    The detailed setting of the video processing pipeline is determined by
    the configuration of the internal batch processing backend. The 
    configuration can be adjusted as a whole by:
      1/ loading a new configuration ("Load Config") from a .json file
      2/ reseting to an internal default ("Reset Config")
    Furthermore, the current configuration can be saved as a .json file for 
    future use via "Save Config".
    
    The configuration (henceforth "config" in short) can be considered as
    consisting of multiple "sections", each contain a number of "options". 
    The current config can be viewed at a per-option basis using "Get Option", 
    and can be set at a per-option basis using "Set Option".
    
    The config consists of 4 mandatory sections termed "modes", "output", 
    "params", and "template_texts". As a general rule, all options within these
    4 sections should be supplied in the config file, EVEN IF the option is
    NOT being used (e.g., "figure_format" in "output" should be set even if 
    there is no figure to be outputted).
    
    The options under the "modes" section are:
      verbosity: (0, 1, or 2) how verbose is the messages printed to stdout 
        while the batch processing happens
      re: (bool) whether the patterns supplied when "RUN!" is clicked are 
        interpreted as regular expression or simple wildcard-enabled string
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
    
    There are 10 options under the "output" section, 5 of which are 
    AUTOMATICALLY set by the user input of filenames in the main GUI when
    "RUN!" is executed:
      details: (bool) whether the detailed motion data is output to csv.
        True if GUI input of "CSV Output" is non-empty; False otherwise 
      summary: (bool) whether the summary of motion data (one line for each
        file) is output to csv. True if GUI input of "Summary Output" is 
        non-empty; False otherwise
      figure: (bool) whether to output (save as file) the main plot of the 
        path taken by the object. True if GUI input of "Figure Output" is 
        non-empty; False otherwise
      video: (bool) whether to generate an overlaid video. True if GUI input 
        of "Video Output" is non-empty; False otherwise
      figure_format: the format (file extension) for which the main plot will
        be saved in. Follow the file extension in the GUI input of "Figure 
        Output"
    The remaining 5 options under the "output" section are:
      record_threshold: (bool) if True, the threshold value will be saved to 
        the summary file
      record_conversion: (bool) if True, the pixel-to-physical unit conversion 
        factor will be asved to the summary file
      record_boxes: (bool) if True, the cropbox, maskbox, and innerbox 
        coordinates will be saved to the summary file
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
    
    There are 14 options under the "params" section. Of these, n_obj is 
    AUTOMATICALLY set by the user input on the number of foreground objects
    in the main GUI when "RUN!" is executed. Of the remaining options, 8 of 
    these have the same meaning whether in interaction mode or not:
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
      reinitialize: whether to reinitialize the initial guess for centroids 
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
    In addition, in interactive mode if both cropbox and maskbox takes the same
      non-None value, the two steps are merged into one where the cropbox is 
      automatically determined as the smallest rectangle enclosing the mask.
    
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
    "export_overlaid_n_video".
    
    Finally, the config may contain a "postprocess" section specifying 
    standard post-processing to be performed after binary frames are obtained.
    The section has 2 options:
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
    '''

    def __init__(self, config_file=None):
        '''
        initialize the main GUI after asking for the configuration file
        '''

        from .batch import MultiObjBatch

        # initialize GUI
        if root is None:
            reinitialize()

        # choose configuration file
        if config_file is None:
            try:
                config = file_chooser(
                    "r", title="Select configuration file..."
                )
            except ValueError as e:
                config = None
        else:
            config = config_file

        # initialize batch processor
        try:
            self.backend = MultiObjBatch(config)

        # show info to user using appropriate message boxes
        except JSONDecodeError as e:
            tkinter.messagebox.showerror(
                title="load_config", 
                message="Invalid .json file. Configuration not loaded"
            )
        else:
            if config is None:
                tkinter.messagebox.showinfo(
                    title="load_config", 
                    message="No file supplied. Default configuration used"
                )
            else:
                tkinter.messagebox.showinfo(
                    title="load_config", 
                    message="Configuration file loaded"
                )

        # initialize GUI
        self.start_frontend()

    def figfile_dialog(self):
        '''
        start a file dialog in which the default extension is obtained live
        from the current backend configuration
        '''
        try:
            # get current output extension
            ext = self.backend.get_option("output", "figure_format")
        except KeyError:
            # file dialog without default extension
            out = tkinter.filedialog.asksaveasfilename()
        else:
            # file dialog with default extension
            out = tkinter.filedialog.asksaveasfilename(
                defaultextension=ext
            )

        # assign result
        self.entries["fig_path"].delete(0, tkinter.END)
        self.entries["fig_path"].insert(tkinter.END, out)

    def start_frontend(self):

        # reinitialize Tk root if none existed
        if root is None:
            reinitialize()

        # create window for main GUI
        frontend = tkinter.Toplevel(master=root)
        frontend.title("Multiple Objects Per-File Processing GUI")

        # creating three organizing frames
        frm_config = tkinter.Frame(master=frontend)
        frm_files = tkinter.Frame(master=frontend)
        frm_prompt = tkinter.Frame(master=frontend)

        # configure the grid of the first frame
        frm_config.columnconfigure(0, weight=0)
        frm_config.columnconfigure(1, weight=0)
        frm_config.columnconfigure(2, weight=0)
        frm_config.columnconfigure(3, weight=1)

        # first frame: two rows of buttons
        btn_load_conf = tkinter.Button(
            master=frm_config, text="Load Config", width=12,
            command=make_filedialog(
                "r", self.load_config_callback
            )
        )
        btn_load_conf.grid(row=0, column=0, padx=3, pady=3)
        btn_save_conf = tkinter.Button(
            master=frm_config, text="Save Config", width=12,
            command=make_filedialog(
                "w", self.save_config_callback, defaultextension=".json"
            )
        )
        btn_save_conf.grid(row=0, column=1, padx=3, pady=3)
        btn_reset_conf = tkinter.Button(
            master=frm_config, text="Reset Config", width=12,
            command=self.reset_config
        )
        btn_reset_conf.grid(row=0, column=2, padx=3, pady=3)
        btn_help = tkinter.Button(
            master=frm_config, text="Help", width=5,
            command=make_message_pop(
                frontend, 80, 15, self.main_help, title="Help", 
                geometry="640x480", preprocess=trim, padx=5, pady=5, 
                bg="white"
            )
        )
        btn_help.grid(row=0, column=3, padx=5, pady=3, sticky="e")
        btn_get_opt = tkinter.Button(
            master=frm_config, text="Get Option", width=12,
            command=self.start_get_GUI
        )
        btn_get_opt.grid(row=1, column=0, padx=3, pady=3)
        btn_set_opt = tkinter.Button(
            master=frm_config, text="Set Option", width=12,
            command=self.start_set_GUI
        )
        btn_set_opt.grid(row=1, column=1, padx=3, pady=3)

        # configure the grid of the second frame
        frm_files.columnconfigure(0, weight=0, minsize=150)
        frm_files.columnconfigure(1, weight=1, minsize=250)

        # right of second frame: entry boxes
        entr_input = tkinter.Entry(master=frm_files)
        entr_input.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        entr_csv = tkinter.Entry(master=frm_files)
        entr_csv.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        entr_fig = tkinter.Entry(master=frm_files)
        entr_fig.grid(row=2, column=1, padx=5, pady=5, sticky="we")
        entr_vid = tkinter.Entry(master=frm_files)
        entr_vid.grid(row=3, column=1, padx=5, pady=5, sticky="we")
        entr_smry = tkinter.Entry(master=frm_files)
        entr_smry.grid(row=4, column=1, padx=5, pady=5, sticky="we")

        # left of second frame: buttons
        btn_input = tkinter.Button(
            master=frm_files, text="Select Input File", width=20,
            command=make_filedialog(
                "r", make_elem_insert(
                    entr_input, tkinter.END, clear=True
                )
            )
        )
        btn_input.grid(row=0, column=0, padx=5, pady=5)
        btn_csv = tkinter.Button(
            master=frm_files, text="Select CSV Output", width=20,
            command=make_filedialog(
                "w", make_elem_insert(
                    entr_csv, tkinter.END, clear=True
                ), defaultextension=".csv"
            )
        )
        btn_csv.grid(row=1, column=0, padx=5, pady=5)
        btn_fig = tkinter.Button(
            master=frm_files, text="Select Figure Output", width=20,
            command=self.figfile_dialog
        )
        btn_fig.grid(row=2, column=0, padx=5, pady=5)
        btn_vid = tkinter.Button(
            master=frm_files, text="Select Video Output", width=20,
            command=make_filedialog(
                "w", make_elem_insert(
                    entr_vid, tkinter.END, clear=True
                ), defaultextension=".mp4"
            )
        )
        btn_vid.grid(row=3, column=0, padx=5, pady=5)
        btn_smry = tkinter.Button(
            master=frm_files, text="Select Summary Output", width=20,
            command=make_filedialog(
                "w", make_elem_insert(
                    entr_smry, tkinter.END, clear=True
                ), defaultextension=".csv"
            )
        )
        btn_smry.grid(row=4, column=0, padx=5, pady=5)

        # third frame: one label + entry box and two buttons
        lbl_nobj = tkinter.Label(
            master=frm_prompt, text="Number of foreground objects:", width=25
        )
        lbl_nobj.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        entr_nobj = tkinter.Entry(master=frm_prompt, width=5)
        entr_nobj.grid(row=0, column=1, padx=5, pady=5)
        try:
            out = self.backend.config["params"]["n_obj"]
        except KeyError:
            out = 2
        entr_nobj.insert(tkinter.END, out)
        btn_run = tkinter.Button(
            master=frm_prompt, text="RUN!", width=5,
            command=self.process
        )
        btn_run.grid(row=0, column=2, padx=10, pady=5)
        btn_exit = tkinter.Button(
            master=frm_prompt, text="Exit", width=5,
            command=self.kill_frontend
        )
        btn_exit.grid(row=0, column=3, padx=10, pady=5)

        # implicit fourth frame: output text box
        outbox = tkinter.scrolledtext.ScrolledText(
            master=frontend, relief="ridge", width=80, height=5, 
            padx=5, pady=5, bg="white"
        )
        outbox.configure(state='disabled')

        # pack the organizing frames to the windows
        frm_config.pack(fill=tkinter.BOTH, expand=False)
        frm_files.pack(fill=tkinter.BOTH, expand=False)
        frm_prompt.pack(fill=tkinter.BOTH, expand=False)
        outbox.pack(fill=tkinter.BOTH, expand=True)

        # define protocol when user closed the GUI window
        frontend.protocol("WM_DELETE_WINDOW", self.kill_frontend)

        # "launch" the main GUI
        frontend.geometry("640x480")
        frontend.update()

        # make sure all sub-windows are initialized
        self.get_GUI = None
        self.set_GUI = None
        self.kill_get_GUI()
        self.kill_set_GUI()

        # hook local variable to instance attributes
        self.frontend = frontend
        self.outbox = outbox
        self.entries = {
            "in_path": entr_input,
            "csv_path": entr_csv,
            "fig_path": entr_fig,
            "vid_path": entr_vid,
            "smry_path": entr_smry,
            "n_obj": entr_nobj
        }

        # start main loop (must be last line!)
        root.mainloop()

    def kill_frontend(self):

        # make sure all children windows are closed
        self.kill_get_GUI()
        self.kill_set_GUI()

        # reset instance attributes
        self.outbox = None
        self.entries = {
            "in_path": None,
            "csv_path": None,
            "fig_path": None,
            "vid_path": None,
            "smry_path": None,
            "n_obj": None
        }
        if self.frontend is not None:
            self.frontend.destroy()
            self.frontend = None

        # kill the current Tk mainloop
        terminate()

    def stdout(self, data, *, position=tkinter.END, end="\n"):

        self.outbox.configure(state='normal')
        self.outbox.insert(position, data + end)
        self.outbox.see(tkinter.END)
        self.outbox.configure(state='disabled')
        self.frontend.update()

    def infer_options(self):
        '''
        set config values based on user's GUI input
        '''

        # alias for easy modifications
        output = self.backend.config["output"]

        raw = [ __.get().strip() for __ in self.entries.values() ]

        if not raw[0]: # empty input file
            tkinter.messagebox.showerror(
                title="RUN!", 
                message="Empty input file. No action taken"
            )
            raise ValueError("Invalid input")

        try:
            n_obj = int(raw[5])
        except ValueError as e:
            tkinter.messagebox.showerror(
                title="RUN!", 
                message="Invalid number of foreground objects. No action taken"
            )
            raise e

        if n_obj < 2:
            tkinter.messagebox.showerror(
                title="RUN!", 
                message=(
                    "Number of foreground objects must be at least 2. " +
                    "No action taken"
                )
            )
            raise ValueError("Invalid input")
        else:
            self.backend.config["params"]["n_obj"] = n_obj

        if raw[1]:
            output["details"] = True
        else: # empty CSV file
            output["details"] = False

        if raw[2]:
            output["figure"] = True
            output["figure_format"] = pathlib.Path(raw[2]).suffix
        else: # empty figure file
            output["figure"] = False

        if raw[3]:
            output["video"] = True
        else: # empty video file
            output["video"] = False

        if raw[4]:
            output["summary"] = True
        else:
            output["summary"] = False

    def process(self):

        # infer config settings from user input
        try:
            self.infer_options()
        except ValueError:
            return

        paths = [ 
            pathlib.Path(__.get().strip()) for __ in self.entries.values() 
        ]

        # infer postprocessing from postprocess section of config
        self.backend.infer_post_ops(clear=True)

        # clear previous texts printed to outbox
        self.outbox.configure(state='normal')
        self.outbox.delete("0.0", tkinter.END)
        self.outbox.configure(state='disabled')

        # route to backend
        self.backend._run([ paths[:4] ], paths[4], stdout=self.stdout)

        # signal to user when all video processing is done
        self.stdout("DONE!")

class MultiObjBatchGUI(BaseBatchGUI):
    '''
    class to implement graphical user interface (GUI) for single object,
    multiple files video motion tracking
    '''

    # help message to be displayed
    main_help = '''
    Graphical user interface (GUI) for multiple objects, multiple files video 
    motion tracking
    
    To run, select the input folder, output folder, and the path of the summary
    file (either through the file dialog accessible via the associated button, 
    or by directly entering the file path into the text box), the pattern to 
    match for input (e.g., "*.mp4" for all .mp4 file), the pattern to 
    substitute as output (NOTE: only meaningful in "re" mode; see below), 
    and the number of foreground objects to detect. Once these are selected, 
    click "RUN!" to start batch processing.
    
    NOTE that while input and output folders are mandatory, the summary file 
    path can be omitted (in which case no summary file is saved). Similarly, 
    both in_pattern and out_pattern can be omitted. If in_pattern is omitted
    all files in in_folder are processed. If out_pattern is omitted in "re" 
    mode, the output files take the same name as the input file except possibly
    for the file extensions. The same convention for output filename also 
    applies in "non-re" mode regardless of out_pattern specification.
    
    The detailed setting of the video processing pipeline is determined by
    the configuration of the internal batch processing backend. The 
    configuration can be adjusted as a whole by:
      1/ loading a new configuration ("Load Config") from a .json file
      2/ reseting to an internal default ("Reset Config")
    Furthermore, the current configuration can be saved as a .json file for 
    future use via "Save Config".
    
    The configuration (henceforth "config" in short) can be considered as
    consisting of multiple "sections", each contain a number of "options".
    The current config can be viewed at a per-option basis using "Get Option", 
    and can be set at a per-option basis using "Set Option".
    
    The config consists of 4 mandatory sections termed "modes", "output", 
    "params", and "template_texts". As a general rule, all options within these
    4 sections should be supplied in the config file, EVEN IF the option is
    NOT being used (e.g., "figure_format" in "output" should be set even if 
    there is no figure to be outputted).
    
    The options under the "modes" section are:
      verbosity: (0, 1, or 2) how verbose is the messages printed to stdout 
        while the batch processing happens
      re: (bool) whether the patterns supplied when "RUN!" is clicked are 
        interpreted as regular expression or simple wildcard-enabled string
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
    
    There are 10 options under the "output" section. Of these, summary is 
    AUTOMATICALLY set by the user input on the summary file (in particular, 
    summary is omitted if the summary file is not set). The remaining 9 
    options are:
      details: (bool) whether the detailed motion data is output to csv
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
    
    There are 14 options under the "params" section. Of these, n_obj is 
    AUTOMATICALLY set by the user input on the number of foreground objects
    in the main GUI when "RUN!" is executed. Of the remaining options, 8 of 
    these have the same meaning whether in interaction mode or not:
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
      reinitialize: whether to reinitialize the initial guess for centroids 
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
    In addition, in interactive mode if both cropbox and maskbox takes the same
      non-None value, the two steps are merged into one where the cropbox is 
      automatically determined as the smallest rectangle enclosing the mask.
    
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
    "export_overlaid_n_video".
    
    Finally, the config may contain a "postprocess" section specifying 
    standard postprocessing to be performed after binary frames are obtained.
    The section has 2 options:
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
    '''

    def __init__(self, config_file=None):
        '''
        initialize the main GUI after asking for the configuration file
        '''

        from .batch import MultiObjBatch

        # initialize GUI
        if root is None:
            reinitialize()

        # choose configuration file
        if config_file is None:
            try:
                config = file_chooser(
                    "r", title="Select configuration file..."
                )
            except ValueError as e:
                config = None
        else:
            config = config_file

        # initialize batch processor
        try:
            self.backend = MultiObjBatch(config)

        # show info to user using appropriate message boxes
        except JSONDecodeError as e:
            tkinter.messagebox.showerror(
                title="load_config", 
                message="Invalid .json file. Configuration not loaded"
            )
        else:
            if config is None:
                tkinter.messagebox.showinfo(
                    title="load_config", 
                    message="No file supplied. Default configuration used"
                )
            else:
                tkinter.messagebox.showinfo(
                    title="load_config", 
                    message="Configuration file loaded"
                )

        # initialize GUI
        self.start_frontend()

    def start_frontend(self):

        # reinitialize Tk root if none existed
        if root is None:
            reinitialize()

        # create window for main GUI
        frontend = tkinter.Toplevel(master=root)
        frontend.title("Multiple Objects Batch Processing GUI")

        # creating three organizing frames
        frm_config = tkinter.Frame(master=frontend)
        frm_files = tkinter.Frame(master=frontend)
        frm_prompt = tkinter.Frame(master=frontend)

        # configure the grid of the first frame
        frm_config.columnconfigure(0, weight=0)
        frm_config.columnconfigure(1, weight=0)
        frm_config.columnconfigure(2, weight=0)
        frm_config.columnconfigure(3, weight=1)

        # first frame: two rows of buttons
        btn_load_conf = tkinter.Button(
            master=frm_config, text="Load Config", width=12,
            command=make_filedialog(
                "r", self.load_config_callback
            )
        )
        btn_load_conf.grid(row=0, column=0, padx=3, pady=3)
        btn_save_conf = tkinter.Button(
            master=frm_config, text="Save Config", width=12,
            command=make_filedialog(
                "w", self.save_config_callback, defaultextension=".json"
            )
        )
        btn_save_conf.grid(row=0, column=1, padx=3, pady=3)
        btn_reset_conf = tkinter.Button(
            master=frm_config, text="Reset Config", width=12,
            command=self.reset_config
        )
        btn_reset_conf.grid(row=0, column=2, padx=3, pady=3)
        btn_help = tkinter.Button(
            master=frm_config, text="Help", width=5,
            command=make_message_pop(
                frontend, 80, 15, self.main_help, title="Help", 
                geometry="640x480", preprocess=trim, padx=5, pady=5, 
                bg="white"
            )
        )
        btn_help.grid(row=0, column=3, padx=5, pady=3, sticky="e")
        btn_get_opt = tkinter.Button(
            master=frm_config, text="Get Option", width=12,
            command=self.start_get_GUI
        )
        btn_get_opt.grid(row=1, column=0, padx=3, pady=3)
        btn_set_opt = tkinter.Button(
            master=frm_config, text="Set Option", width=12,
            command=self.start_set_GUI
        )
        btn_set_opt.grid(row=1, column=1, padx=3, pady=3)

        # configure the grid of the second frame
        frm_files.columnconfigure(0, weight=0, minsize=150)
        frm_files.columnconfigure(1, weight=1, minsize=250)

        # right of second frame (top): entry boxes
        entr_infolder = tkinter.Entry(master=frm_files)
        entr_infolder.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        entr_outfolder = tkinter.Entry(master=frm_files)
        entr_outfolder.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        entr_summary = tkinter.Entry(master=frm_files)
        entr_summary.grid(row=2, column=1, padx=5, pady=5, sticky="we")

        # left of second frame (top): buttons
        btn_infolder = tkinter.Button(
            master=frm_files, text="Select Input Folder", width=20,
            command=make_filedialog(
                "d", make_elem_insert(
                    entr_infolder, tkinter.END, clear=True
                ), mustexist=True
            )
        )
        btn_infolder.grid(row=0, column=0, padx=5, pady=5)
        btn_outfolder = tkinter.Button(
            master=frm_files, text="Select Output Folder", width=20,
            command=make_filedialog(
                "d", make_elem_insert(
                    entr_outfolder, tkinter.END, clear=True
                ), mustexist=True
            )
        )
        btn_outfolder.grid(row=1, column=0, padx=5, pady=5)
        btn_summary = tkinter.Button(
            master=frm_files, text="Select Summary File", width=20,
            command=make_filedialog(
                "w", make_elem_insert(
                    entr_summary, tkinter.END, clear=True
                ), defaultextension=".csv"
            )
        )
        btn_summary.grid(row=2, column=0, padx=5, pady=5)

        # left of second frame (bottom): buttons
        lbl_inpattern = tkinter.Label(
            master=frm_files, text="Input Pattern:", width=20
        )
        lbl_inpattern.grid(row=3, column=0, padx=5, pady=5)
        lbl_outpattern = tkinter.Label(
            master=frm_files, text="Output Pattern:", width=20
        )
        lbl_outpattern.grid(row=4, column=0, padx=5, pady=5)

        # right of second frame (bottom): entry boxes
        entr_inpattern = tkinter.Entry(master=frm_files)
        entr_inpattern.grid(row=3, column=1, padx=5, pady=5, sticky="we")
        entr_outpattern = tkinter.Entry(master=frm_files)
        entr_outpattern.grid(row=4, column=1, padx=5, pady=5, sticky="we")

        # third frame: one label + entry box and two buttons
        lbl_nobj = tkinter.Label(
            master=frm_prompt, text="Number of foreground objects:", width=25
        )
        lbl_nobj.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        entr_nobj = tkinter.Entry(master=frm_prompt, width=5)
        entr_nobj.grid(row=0, column=1, padx=5, pady=5)
        try:
            out = self.backend.config["params"]["n_obj"]
        except KeyError:
            out = 2
        entr_nobj.insert(tkinter.END, out)
        btn_run = tkinter.Button(
            master=frm_prompt, text="RUN!", width=5,
            command=self.process
        )
        btn_run.grid(row=0, column=2, padx=10, pady=5)
        btn_exit = tkinter.Button(
            master=frm_prompt, text="Exit", width=5,
            command=self.kill_frontend
        )
        btn_exit.grid(row=0, column=3, padx=10, pady=5)

        # implicit fourth frame: output text box
        outbox = tkinter.scrolledtext.ScrolledText(
            master=frontend, relief="ridge", width=80, height=5, 
            padx=5, pady=5, bg="white"
        )
        outbox.configure(state='disabled')

        # pack the organizing frames to the windows
        frm_config.pack(fill=tkinter.BOTH, expand=False)
        frm_files.pack(fill=tkinter.BOTH, expand=False)
        frm_prompt.pack(fill=tkinter.BOTH, expand=False)
        outbox.pack(fill=tkinter.BOTH, expand=True)

        # define protocol when user closed the GUI window
        frontend.protocol("WM_DELETE_WINDOW", self.kill_frontend)

        # "launch" the main GUI
        frontend.geometry("640x480")
        frontend.update()

        # make sure all sub-windows are initialized
        self.get_GUI = None
        self.set_GUI = None
        self.kill_get_GUI()
        self.kill_set_GUI()

        # hook local variable to instance attributes
        self.frontend = frontend
        self.outbox = outbox
        self.entries = {
            "in_folder": entr_infolder,
            "out_folder": entr_outfolder,
            "in_pattern": entr_inpattern,
            "out_pattern": entr_outpattern,
            "summary_name": entr_summary,
            "n_obj": entr_nobj
        }

        # start main loop (must be last line!)
        root.mainloop()

    def kill_frontend(self):

        # make sure all children windows are closed
        self.kill_get_GUI()
        self.kill_set_GUI()

        # reset instance attributes
        self.outbox = None
        self.entries = {
            "in_folder": None,
            "out_folder": None,
            "in_pattern": None,
            "out_pattern": None,
            "summary_name": None,
            "n_obj": None
        }
        if self.frontend is not None:
            self.frontend.destroy()
            self.frontend = None

        # kill the current Tk mainloop
        terminate()

    def stdout(self, data, *, position=tkinter.END, end="\n"):

        self.outbox.configure(state='normal')
        self.outbox.insert(position, data + end)
        self.outbox.see(tkinter.END)
        self.outbox.configure(state='disabled')
        self.frontend.update()

    def infer_options(self):
        '''
        set config values based on user's GUI input
        '''

        # infer the value of n_obj in the "params" section
        raw = self.entries["n_obj"].get().strip()
        try:
            n_obj = int(raw)
        except ValueError as e:
            tkinter.messagebox.showerror(
                title="RUN!", 
                message="Invalid number of foreground objects. No action taken"
            )
            raise e

        if n_obj < 2:
            tkinter.messagebox.showerror(
                title="RUN!", 
                message=(
                    "Number of foreground objects must be at least 2. " +
                    "No action taken"
                )
            )
            raise ValueError("Invalid input")
        else:
            self.backend.config["params"]["n_obj"] = n_obj

        # infer the value of summary in the "output" section
        raw = self.entries["summary_name"].get().strip()
        if raw:
            self.backend.config["output"]["summary"] = True
        else:
            self.backend.config["output"]["summary"] = False

    def process(self):

        # infer config settings from user input
        try:
            self.infer_options()
        except ValueError:
            return

        # obtain (string) arguments from the GUI
        args = [ __.get().strip() for __ in self.entries.values() ]
        args = args[:5]

        # check if input/output folders are specified
        if (not args[0]) or (not args[1]):
            tkinter.messagebox.showerror(
                title="RUN!", 
                message=(
                    "Both Input and Output folders must be specified. " +
                    "No action taken"
                )
            )
            return

        # convert empty strings to default values
        if not args[2]: args[2] = None
        if not args[3]: args[3] = None
        if not args[4]: args = args[:4]

        # infer postprocessing from postprocess section of config
        self.backend.infer_post_ops(clear=True)

        # clear previous texts printed to outbox
        self.outbox.configure(state='normal')
        self.outbox.delete("0.0", tkinter.END)
        self.outbox.configure(state='disabled')

        # route operations to the backend
        self.backend(*args, stdout=self.stdout)

__all__ = [
    "root", "file_chooser", 
    "hide", "unhide", "terminate", "reinitialize",
    "MultiObjOneGUI", "MultiObjBatchGUI", 
    "SingleObjOneGUI", "SingleObjBatchGUI"
]

# allow the gui module to be run from command line
if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Batch Video Motion Processing'
    )
    parser.add_argument('-m', '--mode', default="single", help=(
        'object detection mode ("single" or "multi"); ' + 
        'default="single"'
    ))
    parser.add_argument('-f', '--file', default="batch", help=(
        'file selection mode ("batch" or "one"); ' + 
        'default="batch"'
    ))
    parser.add_argument('-c', '--config', default=None, 
        help='configuration .json file')

    args = parser.parse_args()

    mode = args.mode.lower()
    fmode = args.file.lower()
    if mode=="single":
        if fmode=="batch" or fmode=="many":
            gui = SingleObjBatchGUI(args.config)
        elif fmode=="one" or fmode=="per":
            gui = SingleObjOneGUI(args.config)
        else:
            raise ValueError(
                "unknown file selection mode {}".format(args.mode)
            )
    elif mode=="multi":
        if fmode=="batch" or fmode=="many":
            gui = MultiObjBatchGUI(args.config)
        elif fmode=="one" or fmode=="per":
            gui = MultiObjOneGUI(args.config)
        else:
            raise ValueError(
                "unknown file selection mode {}".format(args.mode)
            )
    else:
        raise ValueError("unknown object detection mode {}".format(args.mode))

    # keep the mainloop blocking
    while (gui.frontend is not None) and (root is not None):
        root.mainloop()
