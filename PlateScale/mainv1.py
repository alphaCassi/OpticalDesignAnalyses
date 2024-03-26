from NGS_platescale_optimisation import *

if __name__ == "__main__":

    #Input parameters

    # centroid_noise_std = 10e-6      # centroiding noise in arcsec.
    hole_position_std = 0           # manufacturing tolerance in mm.
    dx = 0.1                        # Shift applied to mask for differential method mm
    dy = 0.1                        # Shift applied to mask for differential method mm
    max_ord = 11                    # max order of polynomial to fit with.
    pin_pitch=0.55                  # distance between pinholes mm

    n_mc = 25                         #Number of noise realizations
    nsamp = 30                        #sampling to evaluate distortion functions over FoV
    # rot_arr = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30]  #field rotation angle
    rot_arr = [0, 30]
    wavenumber = 5                    #wavelength

    poly_orders = np.arange(11,max_ord+1)

    # plate_scale = 0.736 #arcsec/mm
    Rh = 7.95 #m        Metapupil, post-focal DM, radius
    Rg = 4 #m           Ground DM
    r = 4 #m            radius of the EP
    f = 280000 #mm      focal length
    z = 13500 #m         conjugation height of metapupil, it's DM high

    N_ast = 6 #number of asterisms
    n_stars = 3 #number of stars in an asterism

    path = " "    #set your path
    #calling functions
    # stars = NGS_asterisms(N_ast, n_stars, write = False)
    # print(stars[0])
    #plot_asterisms(stars, N_ast)
    # plt.show()
    stars = NGS_asterisms_increaseRad(N_ast, n_stars, image = False, write = False)
    print(stars[0])
    # print(stars.shape)
    # plt.show()

    #obtain input distortions
    # input_dist_ngs_rad, input_dist_pfr_rad = process_distortion(path, stars[:], rot_arr, wavenumber, hole_position_std, pin_pitch)

    #load distortions
    dist_ngs = np.load(path+"input_dist_ngs_rad_60.npy")

    #for the first test. Differences are calculated inside the TT_signal()
    # for i in range(4):
    #     print(f'stars coordinates:\n {stars[i]}')
    #     TT_ngs = TT_signal(stars[i], rot_arr, r, dist_ngs[:, i, :, :], None)
    #     print(f'TT:\n {TT_ngs}')
    # # print(TT_ngs.shape)

    # for the second test
    cv = [command_vector(stars[0], rot_arr, i, Rg, Rh, z, r, dist_ngs[:, 0, :, :], None) for i in range(len(rot_arr))]
    print(f'CV: \n {np.array(cv).squeeze()}')
