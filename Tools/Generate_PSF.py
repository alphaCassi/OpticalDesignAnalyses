import clr, os, winreg
from itertools import islice

# This boilerplate requires the 'pythonnet' module.
# The following instructions are for installing the 'pythonnet' module via pip:
#    1. Ensure you are running a Python version compatible with PythonNET. Check the article "ZOS-API using Python.NET" or
#    "Getting started with Python" in our knowledge base for more details.
#    2. Install 'pythonnet' from pip via a command prompt (type 'cmd' from the start menu or press Windows + R and type 'cmd' then enter)
#
#        python -m pip install pythonnet

fold = 'path\\'
file = 'ZMX.zos'

class PythonStandaloneApplication5(object):
    class LicenseException(Exception):
        pass
    class ConnectionException(Exception):
        pass
    class InitializationException(Exception):
        pass
    class SystemNotPresentException(Exception):
        pass

    def __init__(self, path=None):
        # determine location of ZOSAPI_NetHelper.dll & add as reference
        aKey = winreg.OpenKey(winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER), r"Software\Zemax", 0, winreg.KEY_READ)
        zemaxData = winreg.QueryValueEx(aKey, 'ZemaxRoot')
        NetHelper = os.path.join(os.sep, zemaxData[0], r'ZOS-API\Libraries\ZOSAPI_NetHelper.dll')
        winreg.CloseKey(aKey)
        clr.AddReference(NetHelper)
        import ZOSAPI_NetHelper
        
        # Find the installed version of OpticStudio
        #if len(path) == 0:
        if path is None:
            isInitialized = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize()
        else:
            # Note -- uncomment the following line to use a custom initialization path
            isInitialized = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(path)
        
        # determine the ZOS root directory
        if isInitialized:
            dir = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory()
        else:
            raise PythonStandaloneApplication5.InitializationException("Unable to locate Zemax OpticStudio.  Try using a hard-coded path.")

        # add ZOS-API referencecs
        clr.AddReference(os.path.join(os.sep, dir, "ZOSAPI.dll"))
        clr.AddReference(os.path.join(os.sep, dir, "ZOSAPI_Interfaces.dll"))
        import ZOSAPI

        # create a reference to the API namespace
        self.ZOSAPI = ZOSAPI

        # create a reference to the API namespace
        self.ZOSAPI = ZOSAPI

        # Create the initial connection class
        self.TheConnection = ZOSAPI.ZOSAPI_Connection()

        if self.TheConnection is None:
            raise PythonStandaloneApplication5.ConnectionException("Unable to initialize .NET connection to ZOSAPI")

        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise PythonStandaloneApplication5.InitializationException("Unable to acquire ZOSAPI application")

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise PythonStandaloneApplication5.LicenseException("License is not valid for ZOSAPI use")

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication5.SystemNotPresentException("Unable to acquire Primary system")

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None
        
        self.TheConnection = None
    
    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise PythonStandaloneApplication5.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.LoadFile(filepath, saveIfNeeded)

    def CloseFile(self, save):
        if self.TheSystem is None:
            raise PythonStandaloneApplication5.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise PythonStandaloneApplication5.InitializationException("Unable to acquire ZOSAPI application")

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if self.TheApplication.LicenseStatus == self.ZOSAPI.LicenseStatusType.PremiumEdition:
            return "Premium"
        elif self.TheApplication.LicenseStatus == self.ZOSAPI.LicenseStatusType.EnterpriseEdition:
            return "Enterprise"
        elif self.TheApplication.LicenseStatus == self.ZOSAPI.LicenseStatusType.ProfessionalEdition:
            return "Professional"
        elif self.TheApplication.LicenseStatus == self.ZOSAPI.LicenseStatusType.StandardEdition:
            return "Standard"
        elif self.TheApplication.LicenseStatus == self.ZOSAPI.LicenseStatusType.OpticStudioHPCEdition:
            return "HPC"
        else:
            return "Invalid"
    
    def reshape(self, data, x, y, transpose = False):
        """Converts a System.Double[,] to a 2D list for plotting or post processing
        
        Parameters
        ----------
        data      : System.Double[,] data directly from ZOS-API 
        x         : x width of new 2D list [use var.GetLength(0) for dimension]
        y         : y width of new 2D list [use var.GetLength(1) for dimension]
        transpose : transposes data; needed for some multi-dimensional line series data
        
        Returns
        -------
        res       : 2D list; can be directly used with Matplotlib or converted to
                    a numpy array using numpy.asarray(res)
        """
        if type(data) is not list:
            data = list(data)
        var_lst = [y] * x;
        it = iter(data)
        res = [list(islice(it, i)) for i in var_lst]
        if transpose:
            return self.transpose(res);
        return res
    
    def transpose(self, data):
        """Transposes a 2D list (Python3.x or greater).  
        
        Useful for converting mutli-dimensional line series (i.e. FFT PSF)
        
        Parameters
        ----------
        data      : Python native list (if using System.Data[,] object reshape first)    
        
        Returns
        -------
        res       : transposed 2D list
        """
        if type(data) is not list:
            data = list(data)
        return list(map(list, zip(*data)))

if __name__ == '__main__':
    zos = PythonStandaloneApplication5()
    
    # load local variables
    ZOSAPI = zos.ZOSAPI
    TheApplication = zos.TheApplication
    TheSystem = zos.TheSystem
    TheSystemData = TheSystem.SystemData
    TheAnalyses = TheSystem.Analyses

    TheSystem.LoadFile(fold+file, False)

    # total_wavelengths = TheSystem.SystemData.Wavelengths.NumberOfWavelengths
    wavelengths = [370, 500, 600, 750, 930]
    total_fields = TheSystem.SystemData.Fields.NumberOfFields
    print(total_fields)

    # Loop through each wavelength
    for w, wavelength in enumerate(wavelengths):

        # print(w)
        for a in range(1, total_fields + 1):
            # Create a new Huygens PSF analysis
            new_psf = TheAnalyses.New_HuygensPsf()
            # print(new_psf.HasAnalysisSpecificSettings)

            # Set wavelength and field
            new_psf_settings = new_psf.GetSettings().__implementation__
            # cfgFile = os.environ.get('Temp') + '\\sha.cfg'
            # # Save the current settings to the temp file
            # new_psf_settings.SaveTo(cfgFile)

            new_psf_settings.Wavelength.SetWavelengthNumber(w+1)  #.Wavelength  #keep it w+1 since zemax starts from 1, not 0
            # new_psf_settings.Field = TheSystemData.Fields.GetField(a).FieldNumber
            new_psf_settings.Field.SetFieldNumber(a)
            new_psf_settings.ImageSampleSize = ZOSAPI.Analysis.SampleSizes.S_128x128
            new_psf_settings.PupilSampleSize = ZOSAPI.Analysis.SampleSizes.S_128x128
            new_psf_settings.ImageDelta = round(wavelength * 0.9 / 550, 2)
            # print(new_psf_settings)

            # print(f"Setting Wavelength to: {new_psf_settings.Wavelength}")
            # print(f"Setting Field to: {new_psf_settings.Field}")
            # print(f"Setting ImageSampleSize to: {new_psf_settings.ImageSampleSize}")
            # print(f"Setting PupilSampleSize to: {new_psf_settings.PupilSampleSize}")
            # print(f"Setting ImageDelta to: {new_psf_settings.ImageDelta}")
            # Run the analysis
            new_psf.ApplyAndWaitForCompletion()

            # Save the results
            outfold = r'path\\'
            file_name = f'PSF_Wavelength{wavelength}_Field{a}_edges.txt'
            new_psf_results = new_psf.GetResults()
            new_psf.ToFile(outfold + file_name, showSettings = True)

            new_psf.Close()

    
    
    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # this until you need to.
    del zos
    zos = None
