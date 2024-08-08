# Tropomi-netcdf-data-Processing
 netcdf file variable storage and rasterization.

 
```python
data = collect_data('.nc file')
data = process_data(data)
scan_line_lists, num_scan_lines, data_type =  process_data_to_scan_line_lists(data)

scene = create_matrices(scan_line_lists, num_scan_lines, data_type)
```

You can then call on each scene in created

```python

scene[0]['xch4_corrected']

```