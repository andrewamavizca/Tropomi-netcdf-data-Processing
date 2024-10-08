
# Tropomi-netcdf-data-Processing

 This is a documentation of my progress replicating the journal paper "Automated Detection and monitorig of methane super-emitters using sattelite data". I will be documenting the steps taken to replicate the work in as much 
 detail as possible.
 Link to an article I wrote going over the sliding window implementation on the tropomi data [Medium](https://medium.com/@aamavizca/tropomi-netcdf-file-data-to-raster-while-applying-moving-window-algorithm-33d997a067ca)
## Downloading Data
 The TROPOMI SRON_CH4 data product can be  downloaded here [SRON_CH4](https://ftp.sron.nl/open-access-data-2/TROPOMI/tropomi/ch4/18_17/)
 
 To download each file I wrote a python script that used multiprocessing and then stored them. By referencing the date in the file I put together a text file
 where i recorded the file name with the date and time it corresponded with.
 
 Each file corresponds to one orbit of its daily global coverage. Once the data had been stored and organized by year in Tropomi/18_17/year/file

## Data Processing
 I began processing the data as described in the paper. 

 First step was to organize and store the variables. Which is done in the [collect_data()](Tropomi/my_module/data_processing.py) function. 
 
 The data is then filtered in [process_data()](Tropomi/my_module/data_processing.py) function using the described parameters in the journal paper. 

### Filtering
 Albedo-bias-corrected data with:
  - quality assurance value (QA) >= 0.4
  - methane precision < 10 ppb
  - SWIR aerosol optical depth < 0.13
  - NIR aerosol optical depth < 0.30
  - SWIR surface albedo > 0.02
  - mixed albedo (2.4*NIR surface albedo - 1.13 * SWIR surface albedo) < 0.95
  - SWIR cloud fraction < 0.02
    
  ```python
    def process_data(self):
        non_masked_indices = np.where(~self.data['pixel_id'])[0]

        swir_aerosol_optical_depth = self.data['aerosol_optical_thickness'][:, 1]
        nir_aerosol_optical_depth = self.data['aerosol_optical_thickness'][:, 0]
        
        swir_surface_albedo = self.data['surface_albedo'][:, 1]
        nir_surface_albedo = self.data['surface_albedo'][:, 0]
        
        mixed_albedo = 2.4 * nir_surface_albedo[non_masked_indices] - 1.13 * swir_surface_albedo[non_masked_indices]

        filtered_indices = non_masked_indices[
            (self.data['qa_value'][non_masked_indices] >= 0.4) &
            (self.data['xch4_precision'][non_masked_indices] < 10) &
            (swir_aerosol_optical_depth[non_masked_indices] < 0.13) &
            (nir_aerosol_optical_depth[non_masked_indices] < 0.30) &
            (swir_surface_albedo[non_masked_indices] > 0.02) &
            (mixed_albedo < 0.95) &
            (self.data['cloud_fraction'][non_masked_indices][:, 0] < 0.02)
        ]
        processed_data = {key: self.data[key][filtered_indices] for key in self.data.keys() if key not in [
                       'title','institution', 'source', 'date_created',
                       'l1b_file','pixel_id', 'glintflag','landflag', 'error_id','qa_value','processing_quality_flags']}
        
        return processed_data
   ```
 ### Creating the scenes

- In order to train the model to recognize the scenes, they created a dataset consisting of 32x32 scenes over locations of persistent emissions.
- to gather the 32 x32 scenes a moving algorithim is applied, ensuring that if a plume is on the edge of one scene it would be captured in the center of the next sccene

To begin the moving window algorithim we first organize the data into arrays by its scan line, and indexed by its ground pixel value. [data2slarray()](Tropomi/my_module/data_processing.py) 
Once this is complete the 32x32 windows are "slid" over these arrays 32 scanlines and 32 indices at a time with a 50% overlap on the previous. The 50% overlap is also maintined vertically. Once the window has reached the end of the scan line arrays. 

The following is looped over each file to create a replication of the dataset but now stored as .npy objects with the shape of (# of scenes, # variables, 32, 32). 

```python
data = collect_data('.nc file')
data = process_data(data)
scan_line_lists, num_scan_lines, data_type =  process_data_to_scan_line_lists(data)

scene = create_matrices(scan_line_lists, num_scan_lines, data_type)
```
### Locating Persistent Emissions

We find Regions of interest and then search loop through all the files to see how many scenes overlap with our ROI and then stack those scenes together to later manually sort through them to begin curating our dataset of plumes. 




