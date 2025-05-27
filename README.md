# Optical Design Analyses

## Overview

Welcome to the Optical Design Analyses project repository. This repository aims to provide tools and scripts for conducting thorough analyses in optical design. From understanding wavefront errors to optimizing optical systems, this project seeks to assist researchers, engineers, and enthusiasts in various fields where optical systems play a crucial role.

## Current Status

**MSF_simulation** 

  Pipeline to analyse distortions originating from mid-spatial frequency errors of the optics. See details: *Rebrysh et al. 2025.*

**PlateScale** 

  Pipeline to analyse plate scale variation in the instrument. See details: *Rebrysh et al. 2024.*

**Tools**

 Scripts useful for the analysis of the optical system.

- RMS Wavefront Error (WFE) Processing: This script allows for the processing and plotting of Root Mean Square (RMS) wavefront errors. Wavefront error analysis is essential for evaluating the optical performance of systems, and the provided script facilitates this analysis.

- Zernike Polynomials Calculation for Elliptical Aperture (NEW VERSION IN ao_tools package): Another script is dedicated to calculating Zernike polynomials for an elliptical aperture. Zernike polynomials are widely used in optical design for representing aberrations, and their calculation for non-circular apertures like ellipses is particularly valuable for many practical applications.
We are actively working on developing new features and expanding the capabilities of this repository. Your feedback, contributions, and suggestions are highly encouraged as we continue to grow and improve this project.

- Generation of the PSF with a standalone application with Zemax: the scripts requires a path to and a filename of the Zemax file with the system to generate the Huygens PSF.

- Full analysis of the spectral data of the dichroic coating. It includes transmission/reflection calculation, optical density, transition region, fitting of a notch etc..

Thank you for your interest in Optical Design Analyses. We look forward to building a comprehensive suite of tools to support your optical design endeavors.
