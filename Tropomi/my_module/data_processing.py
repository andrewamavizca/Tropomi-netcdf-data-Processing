from netCDF4 import Dataset
import numpy as np
import warnings

class MethaneDataProcessor:
    def __init__(self, file_name, bbox=None):
        self.file_name = file_name
        self.bbox = bbox
        self.data = self.collect_data()
        self.processed_data = self.process_data()
        self.slarray_data = self.data2slarray()
        self.scenes = self.generate_scenes()

    def collect_data(self):
        if not self.file_name.endswith('.nc'):
            raise ValueError("File must have a .nc extension")
        
        nc = Dataset(self.file_name, 'r')
        data = {
            'title': nc.title,
            'institution': nc.institution,
            'source': nc.source,
            'date_created': nc.date_created,
            'l1b_file': nc.groups['instrument']['l1b_file'][:].data,
            'pixel_id': nc.groups['instrument']['pixel_id'][:].data,
            'scanline': nc.groups['instrument']['scanline'][:].data,
            'ground_pixel': nc.groups['instrument']['ground_pixel'][:].data,
            'time': nc.groups['instrument']['time'][:].data,
            'solar_zenith_angle': nc.groups['instrument']['solar_zenith_angle'][:].data,
            'viewing_zenith_angle': nc.groups['instrument']['viewing_zenith_angle'][:].data,
            'relative_azimuth_angle': nc.groups['instrument']['relative_azimuth_angle'][:].data,
            'latitude_center': nc.groups['instrument']['latitude_center'][:].data,
            'longitude_center': nc.groups['instrument']['longitude_center'][:].data,
            'latitude_corners': nc.groups['instrument']['latitude_corners'][:].data,
            'longitude_corners': nc.groups['instrument']['longitude_corners'][:].data,
            'glintflag': nc.groups['instrument']['glintflag'][:].data,
            'altitude_levels': nc.groups['meteo']['altitude_levels'][:].data,
            'surface_altitude': nc.groups['meteo']['surface_altitude'][:].data,
            'surface_altitude_stdv': nc.groups['meteo']['surface_altitude_stdv'][:].data,
            'dp': nc.groups['meteo']['dp'][:].data,
            'surface_pressure': nc.groups['meteo']['surface_pressure'][:].data,
            'dry_air_subcolumns': nc.groups['meteo']['dry_air_subcolumns'][:].data,
            'landflag': nc.groups['meteo']['landflag'][:].data,
            'u10': nc.groups['meteo']['u10'][:].data,
            'v10': nc.groups['meteo']['v10'][:].data,
            'fluorescence_apriori': nc.groups['meteo']['fluorescence_apriori'][:].data,
            'cloud_fraction': nc.groups['meteo']['cloud_fraction'][:].data,
            'weak_h2o_column': nc.groups['meteo']['weak_h2o_column'][:].data,
            'strong_h2o_column': nc.groups['meteo']['strong_h2o_column'][:].data,
            'weak_ch4_column': nc.groups['meteo']['weak_ch4_column'][:].data,
            'strong_ch4_column': nc.groups['meteo']['strong_ch4_column'][:].data,
            'cirrus_reflectance': nc.groups['meteo']['cirrus_reflectance'][:].data,
            'stdv_h2o_ratio': nc.groups['meteo']['stdv_h2o_ratio'][:].data,
            'stdv_ch4_ratio': nc.groups['meteo']['stdv_ch4_ratio'][:].data,
            'xch4': nc.groups['target_product']['xch4'][:].data,
            'xch4_precision': nc.groups['target_product']['xch4_precision'][:].data,
            'xch4_column_averaging_kernel': nc.groups['target_product'][
                                                      'xch4_column_averaging_kernel'][:].data,
            'ch4_profile_apriori': nc.groups['target_product']['ch4_profile_apriori'][:].data,
            'xch4_apriori': nc.groups['target_product']['xch4_apriori'][:].data,
            'xch4_corrected': nc.groups['target_product']['xch4_corrected'][:].data,
            'fluorescence': nc.groups['side_product']['fluorescence'][:].data,
            'co_column': nc.groups['side_product']['co_column'][:].data,
            'co_column_precision': nc.groups['side_product']['co_column_precision'][:].data,
            'h2o_column': nc.groups['side_product']['h2o_column'][:].data,
            'h2o_column_precision': nc.groups['side_product']['h2o_column_precision'][:].data,
            'spectral_shift': nc.groups['side_product']['spectral_shift'][:].data,
            'aerosol_size': nc.groups['side_product']['aerosol_size'][:].data,
            'aerosol_size_precision': nc.groups['side_product']['aerosol_size_precision'][:].data,
            'aerosol_column': nc.groups['side_product']['aerosol_column'][:].data,
            'aerosol_column_precision': nc.groups['side_product']['aerosol_column_precision'][:].data,
            'aerosol_altitude': nc.groups['side_product']['aerosol_altitude'][:].data,
            'aerosol_altitude_precision': nc.groups['side_product'][
                                                    'aerosol_altitude_precision'][:].data,
            'aerosol_optical_thickness': nc.groups['side_product']['aerosol_optical_thickness'][:].data,
            'surface_albedo': nc.groups['side_product']['surface_albedo'][:].data,
            'surface_albedo_precision': nc.groups['side_product']['surface_albedo_precision'][:].data,
            'reflectance_max': nc.groups['side_product']['reflectance_max'][:].data,
            'processing_quality_flags': nc.groups['diagnostics']['processing_quality_flags'][:].data,
            'convergence': nc.groups['diagnostics']['convergence'][:].data,
            'error_id': nc.groups['diagnostics']['error_id'][:].data,
            'iterations': nc.groups['diagnostics']['iterations'][:].data,
            'chi_squared': nc.groups['diagnostics']['chi_squared'][:].data,
            'chi_squared_band': nc.groups['diagnostics']['chi_squared_band'][:].data,
            'number_of_spectral_points_in_retrieval': nc.groups['diagnostics']['number_of_spectral_points_in_retrieval'][:].data,
            'degrees_of_freedom': nc.groups['diagnostics']['degrees_of_freedom'][:].data,
            'degrees_of_freedom_ch4': nc.groups['diagnostics']['degrees_of_freedom_ch4'][:].data,
            'degrees_of_freedom_aerosol': nc.groups['diagnostics'][
                                                    'degrees_of_freedom_aerosol'][:].data,
            'signal_to_noise_ratio': nc.groups['diagnostics']['signal_to_noise_ratio'][:].data,
            'rms': nc.groups['diagnostics']['rms'][:].data,
            'qa_value': nc.groups['diagnostics']['qa_value'][:].data
        }
        nc.close()
        return data

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
        processed_data = {key: self.data[key][filtered_indices] for key in self.data.keys() if key not in ['title','institution', 'source', 'date_created', 'l1b_file','pixel_id', 'glintflag','landflag', 'error_id','qa_value','processing_quality_flags']}
        
        return processed_data

    def data2slarray(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        variable_info = [
            ('xch4_corrected', 'f4', ()),
            ('latitude_corners', 'f4', (4,)),
            ('longitude_corners', 'f4', (4,)),
            ('u10', 'f4', ()),
            ('v10', 'f4', ()),
            ('latitude_center', 'f4', ()),
            ('longitude_center', 'f4', ()),
            ('scanline', 'i4', ()),
            ('ground_pixel', 'i4', ()),
            ('time', 'i4', (7,)),
            ('solar_zenith_angle', 'f4', ()),
            ('viewing_zenith_angle', 'f4', ()),
            ('relative_azimuth_angle', 'f4', ()),
            ('altitude_levels', 'f4', (13,)),
            ('surface_altitude', 'f4', ()),
            ('surface_altitude_stdv', 'f4', ()),
            ('dp', 'f4', ()),
            ('surface_pressure', 'f4', ()),
            ('dry_air_subcolumns', 'f4', (12,)),
            ('fluorescence_apriori', 'f4', ()),
            ('cloud_fraction', 'f4', (4,)),
            ('weak_h2o_column', 'f4', ()),
            ('strong_h2o_column', 'f4', ()),
            ('weak_ch4_column', 'f4', ()),
            ('strong_ch4_column', 'f4', ()),
            ('cirrus_reflectance', 'f4', ()),
            ('stdv_h2o_ratio', 'f4', ()),
            ('stdv_ch4_ratio', 'f4', ()),
            ('xch4', 'f4', ()),
            ('xch4_precision', 'f4', ()),
            ('xch4_column_averaging_kernel', 'f4', (12,)),
            ('ch4_profile_apriori', 'f4', (12,)),
            ('xch4_apriori', 'f4', ()),
            ('fluorescence', 'f4', ()),
            ('co_column', 'f4', ()),
            ('co_column_precision', 'f4', ()),
            ('h2o_column', 'f4', ()),
            ('h2o_column_precision', 'f4', ()),
            ('spectral_shift', 'f4', (2,)),
            ('aerosol_size', 'f4', ()),
            ('aerosol_size_precision', 'f4', ()),
            ('aerosol_column', 'f4', ()),
            ('aerosol_column_precision', 'f4', ()),
            ('aerosol_altitude', 'f4', ()),
            ('aerosol_altitude_precision', 'f4', ()),
            ('aerosol_optical_thickness', 'f4', (2,)),
            ('surface_albedo', 'f4', (2,)),
            ('surface_albedo_precision', 'f4', (2,)),
            ('reflectance_max', 'f4', (2,)),
            ('convergence', 'i4', ()),
            ('iterations', 'i4', ()),
            ('chi_squared', 'f4', ()),
            ('chi_squared_band', 'f4', (2,)),
            ('number_of_spectral_points_in_retrieval', 'i4', (2,)),
            ('degrees_of_freedom', 'f4', ()),
            ('degrees_of_freedom_ch4', 'f4', ()),
            ('degrees_of_freedom_aerosol', 'f4', ()),
            ('signal_to_noise_ratio', 'f4', (2,)),
            ('rms', 'f4', ())
        ]

        dtype = [(name, dtype, shape) for name, dtype, shape in variable_info]

        max_scan_line = max(self.processed_data['scanline'])
        min_scan_line = min(self.processed_data['scanline'])
        max_ground_pixel = max(self.processed_data['ground_pixel'])
        min_ground_pixel = min(self.processed_data['ground_pixel'])
        num_scan_lines = max_scan_line - min_scan_line + 1
        num_ground_pixels = max_ground_pixel - min_ground_pixel + 1


        # Initialize the structured array
        scan_line_lists = np.full((num_scan_lines, num_ground_pixels), np.nan, dtype=dtype)
        
        # Populate the array
        for idx in range(len(self.processed_data['xch4_corrected'])):
            scan_idx = self.processed_data['scanline'][idx] - min_scan_line
            pixel_idx = self.processed_data['ground_pixel'][idx] - min_ground_pixel

            # Ensure indices are within bounds
            if scan_idx < num_scan_lines and pixel_idx < num_ground_pixels:
                scan_line_lists[scan_idx, pixel_idx] = (
                    #self.processed_data['l1b_file'][idx],
                    self.processed_data['xch4_corrected'][idx],
                    self.processed_data['latitude_corners'][idx],
                    self.processed_data['longitude_corners'][idx],
                    self.processed_data['u10'][idx],
                    self.processed_data['v10'][idx],
                    self.processed_data['latitude_center'][idx],
                    self.processed_data['longitude_center'][idx],
                    self.processed_data['scanline'][idx],
                    self.processed_data['ground_pixel'][idx],
                    self.processed_data['time'][idx],
                    self.processed_data['solar_zenith_angle'][idx],
                    self.processed_data['viewing_zenith_angle'][idx],
                    self.processed_data['relative_azimuth_angle'][idx],
                    self.processed_data['altitude_levels'][idx],
                    self.processed_data['surface_altitude'][idx],
                    self.processed_data['surface_altitude_stdv'][idx],
                    self.processed_data['dp'][idx],
                    self.processed_data['surface_pressure'][idx],
                    self.processed_data['dry_air_subcolumns'][idx],
                    self.processed_data['fluorescence_apriori'][idx],
                    self.processed_data['cloud_fraction'][idx],
                    self.processed_data['weak_h2o_column'][idx],
                    self.processed_data['strong_h2o_column'][idx],
                    self.processed_data['weak_ch4_column'][idx],
                    self.processed_data['strong_ch4_column'][idx],
                    self.processed_data['cirrus_reflectance'][idx],
                    self.processed_data['stdv_h2o_ratio'][idx],
                    self.processed_data['stdv_ch4_ratio'][idx],
                    self.processed_data['xch4'][idx],
                    self.processed_data['xch4_precision'][idx],
                    self.processed_data['xch4_column_averaging_kernel'][idx],
                    self.processed_data['ch4_profile_apriori'][idx],
                    self.processed_data['xch4_apriori'][idx],
                    self.processed_data['fluorescence'][idx],
                    self.processed_data['co_column'][idx],
                    self.processed_data['co_column_precision'][idx],
                    self.processed_data['h2o_column'][idx],
                    self.processed_data['h2o_column_precision'][idx],
                    self.processed_data['spectral_shift'][idx],
                    self.processed_data['aerosol_size'][idx],
                    self.processed_data['aerosol_size_precision'][idx],
                    self.processed_data['aerosol_column'][idx],
                    self.processed_data['aerosol_column_precision'][idx],
                    self.processed_data['aerosol_altitude'][idx],
                    self.processed_data['aerosol_altitude_precision'][idx],
                    self.processed_data['aerosol_optical_thickness'][idx],
                    self.processed_data['surface_albedo'][idx],
                    self.processed_data['surface_albedo_precision'][idx],
                    self.processed_data['reflectance_max'][idx],
                    self.processed_data['convergence'][idx],
                    self.processed_data['iterations'][idx],
                    self.processed_data['chi_squared'][idx],
                    self.processed_data['chi_squared_band'][idx],
                    self.processed_data['number_of_spectral_points_in_retrieval'][idx],
                    self.processed_data['degrees_of_freedom'][idx],
                    self.processed_data['degrees_of_freedom_ch4'][idx],
                    self.processed_data['degrees_of_freedom_aerosol'][idx],
                    self.processed_data['signal_to_noise_ratio'][idx],
                    self.processed_data['rms'][idx],
                )
            slarrayData = {
                'scan_line_lists': scan_line_lists,
                'num_scan_lines': num_scan_lines,
                'data_type': dtype
            }
        return slarrayData

    def generate_scenes(self):
        matrices = []

        max_indices = len(self.slarray_data['scan_line_lists'][0])

        for start_row in range(0, self.slarray_data['num_scan_lines'], 16):
            if start_row + 32 > self.slarray_data['num_scan_lines']:
                break
            
            for start_col in range(0, max_indices, 16):
                if start_col + 32 > max_indices:
                    break
                
                block = self.slarray_data['scan_line_lists'][start_row:start_row+32, start_col:start_col+32]
                
                if np.count_nonzero(~np.isnan(block['xch4_corrected'])) > 220:
                    if self.bbox:
                        if (np.nanmin(block['latitude_corners']) > self.bbox[0] and np.nanmax(block['latitude_corners']) < self.bbox[1] and
                            np.nanmin(block['longitude_corners']) > self.bbox[2] and np.nanmax(block['longitude_corners']) < self.bbox[3]):
                            matrices.append(block)
                    else:
                        matrices.append(block)
                        
        tensor = np.array(matrices, dtype=self.slarray_data['data_type'])
        
        return tensor

def normalize_methane(matrix):
    mean_ch4 = np.nanmean(matrix)
    std_ch4 = np.nanstd(matrix)
    lower_bound = mean_ch4 - std_ch4
    upper_bound = mean_ch4 + 100 - std_ch4  
    
    matrix = np.where(np.isnan(matrix), 0, matrix)
    
    normalized_matrix = np.where(matrix < lower_bound, 0, matrix)
    normalized_matrix = np.where(matrix > upper_bound, 1, normalized_matrix)  
    in_between = (matrix >= lower_bound) & (matrix <= upper_bound)
    normalized_matrix[in_between] = (matrix[in_between] - lower_bound) / (upper_bound - lower_bound)

    return normalized_matrix