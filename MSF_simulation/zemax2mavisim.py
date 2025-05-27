import numpy as np
import mascado.utility.zemax as zemax
import glob
import os

def zemax2mavisim_distgrid(filename, effl, savefn):
    #Load distortion data from file
    df1 = zemax.load_grid_data_variant('B', filename)
    df1 = df1.dropna()

    fx = np.asarray(df1['x-field'])
    fy = np.asarray(df1['y-field'])
    fmax = np.max([fx,fy])
    hx = fx/fmax
    hy = fy/fmax
    x_real = np.asarray(df1['x-real'])
    y_real = np.asarray(df1['y-real'])

    x_pred = -np.tan(fx*np.pi/180)*effl
    y_pred = -np.tan(fy*np.pi/180)*effl

    header = '#To convert from mm to arcsec, the plate scale is 0.736 arcsec/mm\nField_x(deg)  Field_y(deg) Hx Hy  Predicted_x(mm)  Predicted_y(mm)  Real_x(mm)  Real_y(mm)'
    np.savetxt(savefn, np.transpose(np.asarray([fx,fy,hx,hy,x_pred,y_pred,x_real,y_real])), header=header, fmt='%2.8E', comments='')


effl = 280000 #effective focal length
# rot_arr = [0, 30]
# wavenumber = [1, 2, 3, 4, 5, 6, 7, 8, 9]
wavenumber = [5]

fold = r'path'
file_common_part = f'Dist_grid_SCI_'
pattern = fold + file_common_part + '*.txt'
files = glob.glob(pattern)
for file in files:
    base_name = os.path.basename(file).replace('.txt', '')
    print(base_name)
    zemax2mavisim_distgrid(fold + base_name+'.txt', effl, fold+base_name+'_MAVISIM2.txt' )
