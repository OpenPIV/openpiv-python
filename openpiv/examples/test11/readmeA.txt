Instructions for the analysis of case A : 
Loss of seeding in the core of a tip vortex
26.10.2000 		christian.kaehler@dlr.de

Image A001_1.tif and A001_2.tif were recorded at the DNW-LLF in order
to study experimentally the wake vortex formation behind a transport
aircraft (DLR ALVAST half model) in landing configuration (U=60 m/s,
main flow direction is perpendicular to the light-sheet plane). The
measurement position was 1.64 m behind the wing tip and the field of
view is 170 mm by 140 mm. The images were selected as strong gradients,
loss of image density, and varying particle image sizes are common
problems for many PIV applications in large wind tunnels.  


Camera characteristics:

Type                                	PCO SensiCam, see http://www.pco.de 
Sensor technology              		Progressive Scan
Resolution                          	1280 pixel x1024 pixel.
Pixel   size                       	6.7 micrometer x 6.7 micrometer
Dynamic range                     	12 bits (Peltier cooled) 
Quantum efficiency             		typ 40 % 
Full well capacity                  	25000 e 
Readout noise @ 12.5MHz			7 ... 8 e
			

The reference analysis for this case is :

:ev_IS_size_x                           = 32;
:ev_IS_size_y                           = 32;
:ev_IS_size_unit			= "pixel";
:ev_IS_grid_distance_x                	= 16; 
:ev_IS_grid_distance_y               	= 16; 
:ev_IS_grid_distance_unit		= "pixel"; 
:ev_origin_of_evaluation		= 16, 16;
:ev_origin_of_evaluation_units    	= "pixel";
:ev_IS_offset                        	= 0, 0;
:ev_IS_offset_units                	= "pixel";



The mandatory data to be provided are :

Raw data : A001_team_ref_raw.nc

- the raw displacement field corresponding to the highest correlation peak 
- the value of the highest correlation peak for all interrogation positions 
- the correlation values used for the sub-pixel interpolation at the  
  following locations (see table) ; eg. 3 x 3 neighbourhood
- the whole correlation plane (extra file in tecplot format!) calculated 
  at the following locations (see table)

x (pixels)	y (pixels)
512		640
528		528
800		512
----------------------------------------------------------------------------

