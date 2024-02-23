from mavisim import astromsim
from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np


def NGS_asterisms(N_ast, n_stars, write):
    '''
    The generation of N_ast asterisms of 3 NGS stars.

    Parameters:
    N_ast:      desired number of the asterisms to generate.
    n_stars:    number of stars contained in one asterism.

    Return: stars array of the shape (N_ast, 3, 2)
    '''

    np.random.seed(1)    #to keep random values the same for each running
    def func():
        stars_coord = np.zeros((n_stars, 2))
        count = 0
        while count<n_stars:
            coords  = (np.random.random(2)*120)-60
            # diff = DifferenceStarsPos(coords)
            if np.linalg.norm(coords) <= 55.0 and all(np.linalg.norm(coords - stars_coord[:count], axis=1) > 1):
                stars_coord[count,:] = coords
                count += 1
        return stars_coord

    if N_ast == 1:  # Handle the special case when N_ast is 1
        stars = [func()]
    else:
        stars = [func() for _ in range(N_ast)]
    # stars = [[func() for _ in range(N_ast)]]
    stars = np.array(stars)#.squeeze()
    print(stars.shape)
    # print(stars[0, :, 0])

    if write == True:
        np.save(f'D:\Oleksandra\MAVIS\MSF_Distortion\PlateScale\\test_opt\\NGS_asterisms_set3.npy', stars)

    return stars

def plot_asterisms(stars, N_ast):

    import matplotlib.patches as patches
    from matplotlib.transforms import Affine2D
    import math

    if N_ast == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax = [ax]  # Convert to a list for consistency
    else:
        # Calculate number of rows and columns for subplots
        N_rows = math.ceil(math.sqrt(N_ast))
        N_cols = math.ceil(N_ast / N_rows)

        fig, ax = plt.subplots(N_rows, N_cols, figsize=(14, 14))
        ax = ax.ravel()

        # Remove empty subplots if there are any
        for j in range(N_ast, N_rows * N_cols):
            fig.delaxes(ax[j])

    for i in range(N_ast):
        if N_ast > 1:
            if i >= N_rows * N_cols:  # If there are more asterisms than subplots
                break

        im = ax[i].scatter(stars[i, :, 0], stars[i, :, 1], marker = '*', c = 'orange', s = 70)
        patch_tech = patches.Circle((0, 0), radius = 60, transform = ax[i].transData, fill=False, color='darkblue')
        patch_sci = patches.Circle((0, 0), radius = 15, transform = ax[i].transData, fill=False, color='limegreen')
        im.set_transform(Affine2D().scale(1, 1).translate(0, 0))
        ax[i].add_patch(patch_tech)
        ax[i].add_patch(patch_sci)
        ax[i].set_xlim(-62, 62)
        ax[i].set_ylim(-62, 62)
        ax[i].tick_params(axis = 'x', labelsize = 6, length = 2)
        ax[i].tick_params(axis = 'y', labelsize = 6, length = 2)
        ax[i].set_title(f'NGS asterism #{i+1}')

        # Move the beginning of the axes to the circle center
        ax[i].spines['left'].set_position('center')
        ax[i].spines['bottom'].set_position('center')
        ax[i].spines['right'].set_color('none')
        ax[i].spines['top'].set_color('none')


    plt.tight_layout()
    # plt.savefig(f'D:\Oleksandra\MAVIS\MSF_Distortion\PlateScale\\test_opt\\NGS_asterisms_set3.png', dpi = 600)

    return

def NGS_asterisms_increaseRad(N_ast, n_stars, image, write):

    np.random.seed(1)    # to keep random values the same for each run

    def generate_asterism(radius):
        angle_step = 2 * np.pi / n_stars
        stars_coord = np.zeros((n_stars, 2))
        for i in range(n_stars):
            angle = i * angle_step
            stars_coord[i, 0] = radius * np.cos(angle)
            stars_coord[i, 1] = radius * np.sin(angle)
        return stars_coord

    stars = []
    for i in range(1, N_ast + 1):
        radius = min(9 * i, 55.0)  # Adjusted to increase radius with each asterism but capped at max_radius
        asterism = generate_asterism(radius)
        stars.append(asterism)

    stars = np.array(stars)#.squeeze()

    if image == True:
        import matplotlib.patches as patches
        from matplotlib.transforms import Affine2D
        import math

        if N_ast == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax = [ax]  # Convert to a list for consistency
        else:
            # Calculate number of rows and columns for subplots
            N_rows = math.ceil(math.sqrt(N_ast))
            N_cols = math.ceil(N_ast / N_rows)

            fig, ax = plt.subplots(N_rows, N_cols, figsize=(14, 14))
            ax = ax.ravel()

            # Remove empty subplots if there are any
            for j in range(N_ast, N_rows * N_cols):
                fig.delaxes(ax[j])

        for i in range(N_ast):
            if N_ast > 1:
                if i >= N_rows * N_cols:  # If there are more asterisms than subplots
                    break
            im = ax[i].scatter(stars[i, :, 0], stars[i, :, 1], marker = '*', c = 'orange', s = 70)
            patch_tech = patches.Circle((0, 0), radius = 60, transform = ax[i].transData, fill=False, color='darkblue')
            patch_sci = patches.Circle((0, 0), radius = 15, transform = ax[i].transData, fill=False, color='limegreen')
            im.set_transform(Affine2D().scale(1, 1).translate(0, 0))
            ax[i].add_patch(patch_tech)
            ax[i].add_patch(patch_sci)
            ax[i].set_xlim(-62, 62)
            ax[i].set_ylim(-62, 62)
            ax[i].tick_params(axis = 'x', labelsize = 6, length = 2)
            ax[i].tick_params(axis = 'y', labelsize = 6, length = 2)
            ax[i].set_title(f'NGS asterism #{i+1}')

            # Move the beginning of the axes to the circle center
            ax[i].spines['left'].set_position('center')
            ax[i].spines['bottom'].set_position('center')
            ax[i].spines['right'].set_color('none')
            ax[i].spines['top'].set_color('none')

        plt.tight_layout()
        # plt.savefig(f'D:\Oleksandra\MAVIS\MSF_Distortion\PlateScale\\test_opt\\NGS_asterisms_setrad.png', dpi = 600)

    if write == True:
        np.save(f'D:\Oleksandra\MAVIS\MSF_Distortion\PlateScale\\test_opt\\NGS_asterisms_setrad.npy', stars)

    return stars

# def DifferenceStarsPos(coord):
#     '''
#     Calculate the difference between stars positions on the focal plane.
#     Parameters:
#     coord:  an array with stars' coordinates. It should be 1D in the following way [x1, y1, x2, y2, x3, y3].
#             If there are more than 3 stars this function should be modified.
#
#     Return:
#     diff:   an array with the difference in position between stars.
#     '''
#     L = len(coord)
#     D = [coord[i+2] - coord[i] for i in range(L - 2)] + [coord[i+4] - coord[i] for i in range(L - 4)]
#     D_sq = [d**2 for d in D]
#     S = [D_sq[i] + D_sq[i+1] for i in range(len(D_sq) - 1)]
#     diff = np.sqrt(S)
#     return diff


def stars_new_coord(initial_coord, rot_arr, input_dist_ngs = None, input_dist_pfr = None):
    '''
    Calculation of new coordinates of the NGSs' stars accounting for the field rotation and optical distortions.

    Parameters:
    initial_coord:      an array of the initial coordinates.
    rot_arr:        an array of the field rotation angles.

    Return: an array with new coordinates.
    '''
    # initial_coord = np.array([[14, 15],
    #                  [13.7, 21],
    #                  [34, -5]])

    x0 = initial_coord[:, 0]
    y0 = initial_coord[:, 1]

    n_stars = initial_coord.shape[0]
    stars_new_coord = np.empty((len(rot_arr), n_stars, 2))
    for r, angle in enumerate(rot_arr):
        if input_dist_ngs is not None and input_dist_pfr is not None:
            tt_x = np.mean(input_dist_ngs[r, :, 0] + input_dist_pfr[r, :, 0])
            tt_y = np.mean(input_dist_ngs[r, :, 1] + input_dist_pfr[r, :, 1])
            stars_new_coord[r, :, 0] = initial_coord[:, 0] + input_dist_ngs[r, :, 0] + input_dist_pfr[r, :, 0] #- tt_x
            stars_new_coord[r, :, 1] = initial_coord[:, 1] + input_dist_ngs[r, :, 1] + input_dist_pfr[r, :, 1] #- tt_y

        elif input_dist_ngs is not None and input_dist_pfr is None:
            tt_x = np.mean(input_dist_ngs[r, :, 0])
            tt_y = np.mean(input_dist_ngs[r, :, 1])
            stars_new_coord[r, :, 0] = initial_coord[:, 0] + input_dist_ngs[r, :, 0] #- tt_x
            stars_new_coord[r, :, 1] = initial_coord[:, 1] + input_dist_ngs[r, :, 1] #- tt_y

        elif input_dist_pfr is not None and input_dist_ngs is None:
            tt_x = np.mean(input_dist_pfr[r, :, 0])
            tt_y = np.mean(input_dist_pfr[r, :, 1])
            stars_new_coord[r, :, 0] = initial_coord[:, 0] + input_dist_pfr[r, :, 0] #- tt_x
            stars_new_coord[r, :, 1] = initial_coord[:, 1] + input_dist_pfr[r, :, 1] #- tt_y

        else:
            stars_new_coord[r, :, 0] = initial_coord[:, 0]
            stars_new_coord[r, :, 1] = initial_coord[:, 1]


    # print(f'new coord: {stars_new_coord.shape}')

    # plot test
    # plt.figure()
    # plt.scatter(stars_new_coord[:, 0, 0], stars_new_coord[:, 1, 0])
    # plt.scatter(stars_new_coord[:, 0, 1], stars_new_coord[:, 1, 1])
    # plt.scatter(stars_new_coord[:, 0, 2], stars_new_coord[:, 1, 2])
    # plt.show()

    return stars_new_coord

def process_distortion(path, initial_coord, rot_arr, wavenumber, hole_position_std, pin_pitch):

    n_ast = initial_coord.shape[0]
    n_stars = initial_coord.shape[1]
    input_dist_ngs = np.empty((len(rot_arr), n_ast, n_stars, 2))
    input_dist_pfr = np.empty((len(rot_arr), n_ast, n_stars, 2))
    for r, angle in enumerate(rot_arr):
        # rot_distort_NGS = ascii.read(f"D:\Oleksandra\MAVIS\MSF_Distortion\Results_for_NGS\\Dist_grid_ZPL_03_NGS_rot{angle}wave{wavenumber}_MAVISIM.txt" )
        rot_distort_NGS = ascii.read(f"D:\Oleksandra\MAVIS\MSF_Distortion\Results_for_NGS\\new_for_ps\\Dist_grid_ZPL_03_NGS_rot{angle}wave{wavenumber}_MAVISIM2.txt" )
        # rot_distort_NGS = ascii.read(f'D:\Oleksandra\MAVIS\MSF_Distortion\Results_for_NGS\\from_zemax\\Dist_grid_ZPL_03_NGS_rot{angle}wave{wavenumber}_MAVISIM_test1.txt')
        rot_distort_PFR = ascii.read(f'D:\Oleksandra\MAVIS\MSF_Distortion\Results_MSF_issue2.5\Dist_grids\\Dist_grid_ZPL_03_rot{angle}wave{wavenumber}_MAVISIM.txt')
        #Get distortion map with field rotation (use 11-th order polynomials and zero noise to evaluate a close-to-nominal distortion solution)
        astrom_sim_rot_ngs = astromsim.AstromCalibSimE2E(rot_distort_NGS,
                            hole_position_std=hole_position_std,
                            dx=0.05, dy=0.05,
                            n_poly=11, pin_pitch=pin_pitch)

        astrom_sim_rot_pfr = astromsim.AstromCalibSimE2E(rot_distort_PFR,
                            hole_position_std=hole_position_std,
                            dx=0.05, dy=0.05,
                            n_poly=11, pin_pitch=pin_pitch)

        for i in range(n_ast):
            coord = initial_coord[i, :, :]
            # interpolate input distortions at xx/yy coordinates
            input_dist_ngs[r, i, :, 0], input_dist_ngs[r, i, :, 1] = astrom_sim_rot_ngs.input_dist(coord[:, 0], coord[:, 1])
            input_dist_pfr[r, i, :, 0], input_dist_pfr[r, i, :, 1] = astrom_sim_rot_pfr.input_dist(coord[:, 0], coord[:, 1])

    np.save(path+"input_dist_ngs_rad.npy", input_dist_ngs)
    np.save(path+"input_dist_pfr_rad.npy", input_dist_pfr)

    return input_dist_ngs, input_dist_pfr


def TT_signal(initial_coord, rot_arr, r, input_dist_ngs, input_dist_pfr):
    '''
    Calculation of the TT signal.

    Parameters:
    initial_coord:      an array of the initial coordinates.
    rot_arr:        an array of the field rotation angles.
    plate_scale:        a plate scale value.

    Return: an array of TT RMS signal [nm RMS]
    '''

    # x0 = initial_coord[:, 0]
    # y0 = initial_coord[:, 1]

    strs_coord = stars_new_coord(initial_coord, rot_arr, input_dist_ngs, input_dist_pfr)
    x0 = strs_coord[0, :, 0]
    y0 = strs_coord[0, :, 1]

    # Vectorized approach for RMS calculations
    diff_x = (strs_coord[:, :, 0] - x0) / 1.25
    diff_y = (strs_coord[:, :, 1] - y0) / 1.25

    print(f'deltas:\n{np.stack((diff_x, diff_y), axis = -1)}')

    # Apply the formula in a vectorized manner
    rms_x = 2*r * 1e+9 * np.arctan(diff_x / 164800) / 4
    rms_y = 2*r * 1e+9 * np.arctan(diff_y / 164800) / 4
    # rms_x = np.arctan(diff_x / 164800)*206265*1e6  #TT in uas
    # rms_y = np.arctan(diff_y / 164800)*206265*1e6

    # Stack and reshape the results to match the desired output format
    rms_tt_allrot = np.stack((rms_x, rms_y), axis=-1)

    return rms_tt_allrot


def interaction_matrix(initial_coord, Rg, Rh, z, r):
    '''
    Compute the interaction matrix.

    Parameters:
    initial_coord:      an array of the initial coordinates.
    Rg:         a radius of the ground DM.
    Rh:         a radius of the high DM.
    r:          a radius of the subaperture.

    Return:
    '''

    # initial_coord = np.array([[14, 15],
    #                  [13.7, 21],
    #                  [34, -5]])
    x = initial_coord[:, 0]
    y = initial_coord[:, 1]
    h_x = z* np.tan(x/206265)
    h_y = z* np.tan(y/206265)
    h = np.sqrt(h_x**2 + h_y**2)
    alpha = np.arctan(h_y/h_x)
    # print(h)
    # print(alpha)
    tt = r/Rg #tt coefficient

    f11 = 2*np.sqrt(3)*h[0]*r*np.cos(alpha[0])/Rh**2    #focus coefficients
    f12 = 2*np.sqrt(3)*h[0]*r*np.sin(alpha[0])/Rh**2
    f21 = 2*np.sqrt(3)*h[1]*r*np.cos(alpha[1])/Rh**2
    f22 = 2*np.sqrt(3)*h[1]*r*np.sin(alpha[1])/Rh**2
    f31 = 2*np.sqrt(3)*h[2]*r*np.cos(alpha[2])/Rh**2
    f32 = 2*np.sqrt(3)*h[2]*r*np.sin(alpha[2])/Rh**2

    ast4_11 = np.sqrt(6)*h[0]*r*np.sin(alpha[0])/Rh**2    #astigmatism coefficients
    ast4_12 = np.sqrt(6)*h[0]*r*np.cos(alpha[0])/Rh**2
    ast4_21 = np.sqrt(6)*h[1]*r*np.sin(alpha[1])/Rh**2
    ast4_22 = np.sqrt(6)*h[1]*r*np.cos(alpha[1])/Rh**2
    ast4_31 = np.sqrt(6)*h[2]*r*np.sin(alpha[2])/Rh**2
    ast4_32 = np.sqrt(6)*h[2]*r*np.cos(alpha[2])/Rh**2
    ast5_11 = np.sqrt(6)*h[0]*r*np.cos(alpha[0])/Rh**2
    ast5_12 = -np.sqrt(6)*h[0]*r*np.sin(alpha[0])/Rh**2
    ast5_21 = np.sqrt(6)*h[1]*r*np.cos(alpha[1])/Rh**2
    ast5_22 = -np.sqrt(6)*h[1]*r*np.sin(alpha[1])/Rh**2
    ast5_31 = np.sqrt(6)*h[2]*r*np.cos(alpha[2])/Rh**2
    ast5_32 = -np.sqrt(6)*h[2]*r*np.sin(alpha[2])/Rh**2

    IM = np.array([[tt, 0, f11, ast4_11, ast5_11],
                   [0, tt, f12, ast4_12, ast5_12],
                   [tt, 0, f21, ast4_21, ast5_21],
                   [0, tt, f22, ast4_22, ast5_22],
                   [tt, 0, f31, ast4_31, ast5_31],
                   [0, tt, f32, ast4_32, ast5_32]])


    return IM

def reconstruction_matrix(initial_coord, Rg, Rh, z, r):
    '''
    Compute the reconstruction matrix which is the pseudo-inverse interaction matrix.

    Parameters:
    initial_coord:      an array of the initial coordinates.
    Rg:         a radius of the ground DM.
    Rh:         a radius of the high DM.
    # r:          a radius of the subaperture.

    Return: reconstruction matrix
    '''

    IM = interaction_matrix(initial_coord, Rg, Rh, z, r)
    #reconstruction matrix as pseudo-inverse IM
    RM = np.linalg.pinv(IM)

    return RM

def command_vector(initial_coord, rot_arr, rot_arr_number, Rg, Rh, z, r, input_dist_ngs, input_dist_pfr):
    '''
    Computation of the vector of commands that are applied to the DMs.

    Parameters:
    initial_coord:      an array of the initial coordinates.
    rot_arr:        an array of the field rotation angles.
    plate_scale:        a plate scale value.
    Rg:         a radius of the ground DM.
    Rh:         a radius of the high DM.
    r:          a radius of the subaperture.

    Return: command vector
    '''

    RM = reconstruction_matrix(initial_coord, Rg, Rh, z, r)
    TT_signals = TT_signal(initial_coord, rot_arr, r, input_dist_ngs, input_dist_pfr)[rot_arr_number]
    TT_signals_flttnd = TT_signals.flatten()

    command_v = np.dot(RM, TT_signals_flttnd)

    return command_v

def projection_matrix(Rg, Rh, x, y, z, r):
    '''
    Computation of the projection matrix.
    alpha: polar angle. It should be in radians.

    Parameters:
    Rg:         a radius of the ground DM.
    x and y:    coordinates of a star in arcsec

    Return: projection matrix
    '''

    h_x = z* np.tan(x/206265)
    h_y = z* np.tan(y/206265)
    h = np.sqrt(h_x**2 + h_y**2)
    alpha = np.arctan(h_y/h_x)

    tt = r/Rg
    f11 = 2*np.sqrt(3)*h*r*np.cos(alpha)/Rh**2    #focus coefficients
    f12 = 2*np.sqrt(3)*h*r*np.sin(alpha)/Rh**2
    ast4_11 = np.sqrt(6)*h*r*np.sin(alpha)/Rh**2    #astigmatism coefficients
    ast4_12 = np.sqrt(6)*h *r*np.cos(alpha)/Rh**2
    ast5_11 = np.sqrt(6)*h *r*np.cos(alpha)/Rh**2
    ast5_12 = -np.sqrt(6)*h *r*np.sin(alpha)/Rh**2

    PM = np.array([[tt, 0, f11, ast4_11, ast5_11],
                  [0, tt, f12, ast4_12, ast5_12]])

    return PM

def Observed_star_TT(initial_coord, rot_arr, rot_arr_number, Rg, Rh, x, y, z, r, input_dist_ngs, input_dist_pfr):
    '''
    Computation of plate scale signals. Here, they are so-called plate scale modes - stretching in x and y directions. [mm]

    Parameters:
    initial_coord:      an array of the initial coordinates of the WFSs' centroids.
    rot_arr:        an array of the field rotation angles.
    plate_scale:        a plate scale value.
    Rg:         a radius of the ground DM.
    Rh:         a radius of the high DM.
    h:      a radial displacement.
    alpha:  a polar angle. It should be in radians.
    r:          a radius of the subaperture.

    Return: an array of PS signal values.
    '''

    PM = projection_matrix(Rg, Rh, x, y, z, r)
    Com_vec = command_vector(initial_coord, rot_arr, rot_arr_number, Rg, Rh, z, r, input_dist_ngs, input_dist_pfr)

    PS = np.dot(PM, Com_vec)

    return PS

# def observed_stars(N_st):
#     'Create set of stars within the science FoV, for which we want to check PS variations.'
#
#     stars_coord = np.zeros((N_st, 2))
#     count = 0
#     while count<N_st:
#         coords_x  = (np.random.random()*30)-15
#         coords_y = (np.random.random()*30)-15
#         if coords_x <= 14.9 and coords_y <= 14.9:
#             stars_coord[count, 0] = coords_x
#             stars_coord[count, 1] = coords_y
#             count += 1
#
#     # print(stars_coord)
#
#     return stars_coord

def observed_stars(N_st):
    'Create set of stars within the science FoV, for which we want to check PS variations.'

    stars_coord = np.zeros((N_st, 2))
    count = 0
    max_radius = 14.9
    radius_step = max_radius / np.sqrt(N_st)  # Ensure radially increasing distance

    i = 1
    while count < N_st:
        radius = i * radius_step
        angle = np.random.random() * 2 * np.pi

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        stars_coord[count, 0] = x
        stars_coord[count, 1] = y
        count += 1

        i += 1

    return stars_coord







