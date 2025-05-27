from mavisim import astromsim
from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np


static_distort =  ascii.read("filename.txt")  #distortion field used for calibration, wavelength = 550 nm
# static_distort =  ascii.read("D:\Oleksandra\MAVIS\\MSF_Distortion\\Results_for_NGS\\new_for_ps\\Dist_grid_ZPL_03_NGS_rot0wave5_MAVISIM2.txt")

# centroid_noise_std = 10e-6      # centroiding noise in arcsec.
hole_position_std = 0           # manufacturing tolerance in mm.
dx = 0.1                        # Shift applied to mask for differential method mm
dy = 0.1                       # Shift applied to mask for differential method mm
max_ord = 12                   # max order of polynomial to fit with.
pin_pitch=0.55                 # distance between pinholes mm

# n_mc = 25     #Number of noise realizations
nsamp = 40 #30      #sampling to evaluate distortion functions over FoV
rot_arr = [0.26, 2.37, 5.13]
# rot_arr = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]
# wavenumber = [1, 2, 3, 4, 5, 6, 7, 8, 9]
wavenumber = [5]

#---------------- Evaluate astrometric error between objects ------------------

#generate objects
n_objects  = 400
count = 0
objects = np.zeros((n_objects,2))
while count<n_objects:
    coords  = np.random.random(2)*30-15
    if ((coords**2).sum()**0.5)<=14.9:
        objects[count,:] = coords
        count += 1


#Calculate distances between objects
dists = []
for i in range(n_objects):
    for j in range(i):
        ob1 = objects[i,:]
        ob2 = objects[j,:]
        true_distance = np.linalg.norm(ob1-ob2)
        dists.append(true_distance)
dists = np.r_[dists]
dista = np.arange(0,np.max(dists),0.02)

# Values to evaluate distortion functions with. Can be any arangement of
# coordinates within the science field.
xx,yy = np.meshgrid(np.linspace(-15,15,nsamp),np.linspace(-15,15,nsamp))
xx = xx.flatten()
yy = yy.flatten()
valid = (xx**2+yy**2)**0.5<=14.9
xx = xx[valid]
yy = yy[valid]
# rr = (xx**2+yy**2)**0.5
# rr_sq = xx**2+yy**2
# # use slightly smaller field because distortions arent defined beyond 30"x30" box
# maxrad = 14.2  #was 14.9
# xx = xx[rr_sq<=maxrad**2]
# yy = yy[rr_sq<=maxrad**2]


poly_orders = np.arange(11,max_ord+1)
rms_residuals = np.zeros((len(poly_orders), len(wavenumber), len(rot_arr)))
errors_d = np.zeros((len(poly_orders),len(dista), len(wavenumber), len(rot_arr)))


for r, angle in enumerate(rot_arr):
    for w, wavelength in enumerate(wavenumber):
    # for w in range(len(wavenumber)):
        # print(angle)
        # print(wavelength)
        draw = 1
        draw2 = 1
        rot_distort = ascii.read(f'Dist_grid_MSF_NGC3201__{angle}deg_{wavelength}.txt')
        # rot_distort = ascii.read(f'D:\Oleksandra\MAVIS\MSF_Distortion\Results_for_NGS\\new_for_ps\\Dist_grid_ZPL_03_NGS_rot{angle}wave{wavelength}_MAVISIM2.txt')

        astrom_sim_rot = astromsim.AstromCalibSimE2E(rot_distort,
                                                hole_position_std=0.0,
                                                dx=0.05, dy=0.05,
                                                n_poly=12, pin_pitch=pin_pitch)

        input_dist_xx, input_dist_yy = astrom_sim_rot.input_dist(xx, yy)

        # plt.figure(figsize=[5,5])
        # plt.quiver(xx, yy, input_dist_xx, input_dist_yy)
        # plt.title('RMS input [uas]: %2.1f' %(np.sqrt(np.std(input_dist_xx)**2 + np.std(input_dist_xx)**2)*1e6)) #I changed to mas here
        # plt.axis('square')
        # plt.xlabel('X Field (arcsec)')
        # plt.ylabel('Y Field (arcsec)')
        # plt.savefig(f'D:\Oleksandra\MAVIS\MSF_Distortion\Results_for_NGS\\new_for_ps\\distortionmaps\\Input_map_rot{angle}wave{wavelength}_NoNoise_11thOrder_NGS_e2e_specidiedwavelength.png', dpi=300)

        for k in range(len(poly_orders)):
            poly_ord = poly_orders[k]
            print('calculating polynomial order: %d' %poly_ord)
            astrom_sim = astromsim.AstromCalibSimE2E(
                    static_distort,
                    hole_position_std=hole_position_std,
                    dx=dx, dy=dy,
                    n_poly=poly_ord, pin_pitch=pin_pitch)

            recovered_dist_xx,recovered_dist_yy = astrom_sim.recovered_dist(xx,yy)

            # plt.figure(figsize=[5,5])
            # plt.quiver(xx, yy, recovered_dist_xx, recovered_dist_yy)
            # plt.title('RMS recovered [uas]: %2.1f' %(np.sqrt(np.std(recovered_dist_xx)**2 + np.std(recovered_dist_yy)**2)*1e6)) #I changed to mas here
            # plt.axis('square')
            # plt.xlabel('X Field (arcsec)')
            # plt.ylabel('Y Field (arcsec)')
            # plt.savefig(f'D:\Oleksandra\MAVIS\MSF_Distortion\Results_for_NGS\\new_for_ps\\distortionmaps\\Recovered_map_rot{angle}wave{wavelength}_NoNoise_11thOrder_NGS_e2e_specidiedwavelength.png', dpi=300)

            #calculate shift and plate scale correction between calibration map and rotated fov map
            tiltx = np.mean(input_dist_xx-recovered_dist_xx)
            tilty = np.mean(input_dist_yy-recovered_dist_yy)

            cond = xx!=0
            scalex = np.mean((input_dist_xx[cond]-recovered_dist_xx[cond]-tiltx)/xx[cond])
            cond = yy!=0
            scaley = np.mean((input_dist_yy[cond]-recovered_dist_yy[cond]-tilty)/yy[cond])

            #get residuals of calibration after tip-tilt and plate scale correction
            residuals_x = input_dist_xx - recovered_dist_xx - tiltx - xx*scalex
            residuals_y = input_dist_yy - recovered_dist_yy - tilty - yy*scaley

            if draw==1 and poly_ord==12:
                plt.figure(figsize=[5,5])
                plt.quiver(xx,yy,residuals_x,residuals_y)
                plt.title('RMS residuals [uas]: %2.1f' %(np.sqrt(np.mean(residuals_x**2 + residuals_y**2))*1e6))
                plt.axis('square')
                plt.xlabel('X Field (arcsec)')
                plt.ylabel('Y Field (arcsec)')
                draw = 0

            # rms_residuals[k, w, r] = np.sqrt(np.std(residuals_x)**2 + np.std(residuals_y)**2)*1e6
            rms_residuals[k, w, r] = np.sqrt(np.mean(input_dist_xx**2 + input_dist_yy**2))*1e6

            #calculate objects measured positions after tip-tilt and plate scale correction
            obj_input_dis_x, obj_input_dis_y = astrom_sim_rot.input_dist(objects[:,0],objects[:,1])
            obj_recov_dis_x, obj_recov_dis_y = astrom_sim.recovered_dist(objects[:,0],objects[:,1])
            obj_meas_x = objects[:,0] + obj_input_dis_x - obj_recov_dis_x - tiltx - objects[:,0]*scalex
            obj_meas_y = objects[:,1] + obj_input_dis_y - obj_recov_dis_y - tilty - objects[:,1]*scaley
            ob_meas = np.transpose(np.asarray([obj_meas_x,obj_meas_y]))

            m_dists = []

            #Calculate distances between objects
            for i in range(n_objects):
                for j in range(i):
                    meas_ob1 = ob_meas[i,:]
                    meas_ob2 = ob_meas[j,:]
                    meas_distance = np.linalg.norm(meas_ob1-meas_ob2)
                    m_dists.append(meas_distance)
            m_dists = np.r_[m_dists]
            errs  = dists-m_dists

            if draw2==1 and poly_ord==11:
                plt.figure(figsize=[5,5])
                plt.plot(dists,np.abs(errs)*1e6,"b.", markersize=1)
                plt.title("Astrometric Error vs Distance")
                plt.xlabel("Separation [arcsec]")
                plt.ylabel("Astrometric Error after calibration [uas]")
                plt.grid(True, 'major', ls='--', lw=.5, c='k', alpha=.3)
                draw2 = 0

            err_d = np.zeros(len(dista))
            for i in range(len(dista)):
                try:
                    err_d[i] = np.percentile(np.abs(errs[np.where(dists<=dista[i])]),98)
                except:
                    err_d[i] = 0

            errors_d[k,:, w, r] = err_d

# rms_residuals = np.mean(rms_residuals, axis=1)
# errors_d = np.mean(errors_d,axis=2)
























