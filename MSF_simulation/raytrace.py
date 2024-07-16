import clr, os, winreg
from win32com.client import CastTo, constants
from itertools import islice
import numpy as np
from System import Enum, Int32, Double

import time

#*********************************************************************
#------------------------INPUT PARAMETERS-----------------------------
#*********************************************************************

fold = 'your_folder'

file = 'your_file.zos'

pup_samp = 150
field_samp = 40

outfold = 'outfolder\\'
outfile = 'outfile'

rot_arr = [0, 1, 2, 3, 5, 7, 10., 15., 20., 30.]


#*********************************************************************

def do_raytrace(surf, wavenumber, hx, hy, pup_samp):
    ppx = np.linspace(-1,1,pup_samp)
    ppxx, ppyy = np.meshgrid(ppx, ppx)
    max_rays = np.size(ppxx)
    
    raytrace = Tools.OpenBatchRayTrace()
    normUnPolData = raytrace.CreateNormUnpol(max_rays+1, ZOSAPI.Tools.RayTrace.RaysType.Real, surf)         
    for i in range(np.size(ppxx, axis=0)):
        for j in range(np.size(ppxx, axis=1)):            
            normUnPolData.AddRay(wavenumber, hx, hy, ppxx[i,j], ppyy[i,j], Enum.Parse(ZOSAPI.Tools.RayTrace.OPDMode, "None"))
            
    raytrace.RunAndWaitForCompletion()
    raytrace.Close()
    return normUnPolData

def get_centroid(ray_data):
    x_rays = []
    y_rays = []
    ray_data.StartReadingResults()
    
    # Python NET requires all arguments to be passed in as reference, so need to have placeholders
    sysInt = Int32(1)
    sysDbl = Double(1.0)
    
    output = ray_data.ReadNextResult(sysInt, sysInt, sysInt,
                    sysDbl, sysDbl, sysDbl, sysDbl, sysDbl, sysDbl, 
                    sysDbl, sysDbl, sysDbl, sysDbl, sysDbl);
    while output[0]:                                                    # success
        if ((output[2] == 0) and (output[3] == 0)):                     # ErrorCode & vignetteCode
            x_rays.append(output[4])  # X
            y_rays.append(output[5])   # Y
        output = ray_data.ReadNextResult(sysInt, sysInt, sysInt,
                    sysDbl, sysDbl, sysDbl, sysDbl, sysDbl, sysDbl, 
                    sysDbl, sysDbl, sysDbl, sysDbl, sysDbl);
        
    cenx = np.mean(x_rays)
    ceny = np.mean(y_rays)
    return cenx, ceny


class PythonStandaloneApplication1(object):
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
            raise PythonStandaloneApplication1.InitializationException("Unable to locate Zemax OpticStudio.  Try using a hard-coded path.")

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
            raise PythonStandaloneApplication1.ConnectionException("Unable to initialize .NET connection to ZOSAPI")

        self.TheApplication = self.TheConnection.CreateNewApplication()
        if self.TheApplication is None:
            raise PythonStandaloneApplication1.InitializationException("Unable to acquire ZOSAPI application")

        if self.TheApplication.IsValidLicenseForAPI == False:
            raise PythonStandaloneApplication1.LicenseException("License is not valid for ZOSAPI use")

        self.TheSystem = self.TheApplication.PrimarySystem
        if self.TheSystem is None:
            raise PythonStandaloneApplication1.SystemNotPresentException("Unable to acquire Primary system")

    def __del__(self):
        if self.TheApplication is not None:
            self.TheApplication.CloseApplication()
            self.TheApplication = None
        
        self.TheConnection = None
    
    def OpenFile(self, filepath, saveIfNeeded):
        if self.TheSystem is None:
            raise PythonStandaloneApplication1.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.LoadFile(filepath, saveIfNeeded)

    def CloseFile(self, save):
        if self.TheSystem is None:
            raise PythonStandaloneApplication1.SystemNotPresentException("Unable to acquire Primary system")
        self.TheSystem.Close(save)

    def SamplesDir(self):
        if self.TheApplication is None:
            raise PythonStandaloneApplication1.InitializationException("Unable to acquire ZOSAPI application")

        return self.TheApplication.SamplesDir

    def ExampleConstants(self):
        if self.TheApplication.LicenseStatus == self.ZOSAPI.LicenseStatusType.PremiumEdition:
            return "Premium"
        elif self.TheApplication.LicenseStatus == self.ZOSAPI.LicenseStatusTypeProfessionalEdition:
            return "Professional"
        elif self.TheApplication.LicenseStatus == self.ZOSAPI.LicenseStatusTypeStandardEdition:
            return "Standard"
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
    zos = PythonStandaloneApplication1()
    
    # load local variables
    ZOSAPI = zos.ZOSAPI
    TheApplication = zos.TheApplication
    TheSystem = zos.TheSystem
    TheSystemData = TheSystem.SystemData
    TheAnalyses = TheSystem.Analyses
    TheLDE = TheSystem.LDE
    TheMFE = TheSystem.MFE
    TheMCE = TheSystem.MCE
    Tools = TheSystem.Tools
    
    TheSystem.LoadFile(fold+file, False)
    
    # Set up Batch Ray Trace
    nsur = TheSystem.LDE.NumberOfSurfaces
    
    # Determine maximum field in Y only
    num_fields = TheSystem.SystemData.Fields.NumberOfFields
    fmax = 0.0
    for i in range(1, num_fields + 1):
        xf = TheSystem.SystemData.Fields.GetField(i).X
        yf = TheSystem.SystemData.Fields.GetField(i).Y
        if (np.sqrt(xf**2+yf**2) > fmax):
            fmax = np.sqrt(xf**2+yf**2)
    
    # Define batch ray trace constants    
    wavenumber = [5]
    # wavenumber = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    hx = np.linspace(-1,1,field_samp)
    hy = np.linspace(-1,1,field_samp)
    
    time1 = time.time()
    
    for r in range(len(rot_arr)):
    # for r, angle in enumerate(rot_arr):
        TheLDE.GetSurfaceAt(1).GetSurfaceCell(ZOSAPI.Editors.LDE.SurfaceColumn.Par5).DoubleValue=rot_arr[r]
        
        # TheSystem.SaveAs('D:\\Oleksandra\\MAVIS\\MSF_Distortion\\output_NGS.zos')
        # TheLDE.GetSurfaceAt(1).GetCellAt(16).DoubleValue=angle
        # TheLDE.GetSurfaceAt(1).SurfaceData.TiltAbout_Z = angle
        #TheSystem.SaveAs(outfold+'ang%d.zos' %rot_arr[r])
        
        cenx = np.zeros((field_samp, field_samp))
        ceny = np.zeros((field_samp, field_samp))
        for w in range(len(wavenumber)):
            for p in range(len(hx)):
                for q in range(len(hy)):

                    if np.sqrt(hx[p]**2 + hy[q]**2)< 1:
                        rays = do_raytrace(nsur, wavenumber[w], hx[p], hy[q], pup_samp)
                        cenx[p,q], ceny[p,q] = get_centroid(rays)


                        # Read batch raytrace and display results

            time2 = time.time()
            print('Elapsed time: %d seconds' %(time2-time1))

            #write to file
            fid = open(outfold+outfile+'rot%d' %rot_arr[r] + 'wave%d' %wavenumber[w] +'.txt', 'w')

            fid.writelines('Executing PATH\n')
            fid.writelines('start\n')
            fid.writelines('A 0.00000E+00\n')
            fid.writelines('B 0.00000E+00\n')
            fid.writelines('C 0.00000E+00\n')
            fid.writelines('D 0.00000E+00\n')
            fid.writelines('EFFL 0.0000	mm\n')
            fid.writelines('Npoint \t Input_X deg \t Input_Y \t Distorted_X mm \t Distorted_Y\n')

            npoint = 0
            for i in range(np.size(cenx, axis=0)):
                for j in range(np.size(cenx, axis=1)):
                    if np.sqrt(hx[i]**2 + hy[j]**2)< 1:
                        npoint += 1
                        fid.writelines('%2.8e %2.8e %2.8e %2.8e %2.8e\n' %(npoint, hx[i]*fmax, hy[j]*fmax, cenx[i,j], ceny[i,j]))

            fid.close()
    
    # This will clean up the connection to OpticStudio.
    # Note that it closes down the server instance of OpticStudio, so you for maximum performance do not do
    # this until you need to.
    del zos
    zos = None
