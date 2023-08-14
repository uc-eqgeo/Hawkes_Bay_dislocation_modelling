import numpy as np
import pickle as pkl
import os

### set up reusable fault centers
# between kidnappers ridge and Waimarama faults
cen1_x, cen1_y = 1965700, 5598000
# SW kidnapprs ridge
cen2_x, cen2_y = 1969663, 5602945
# fault bounding SE kidnappers basin. Depending on the reference, it's either NW or SE dipping?
cen3_x, cen3_y = 1947170, 5621170

### nomenclature
# lis = listric; plnr = planar
# letter is fault shape option; number is variation (e.g., with or without subduction interface)
# patch heights in meters; dips in degrees

## model A: trace at Kidnappers ridge; ~same shape as Lachlan fault
# depth to planar base: 6 km; horizontal dist to planar base: 5.87 km;
# depth to interface: 21 km; horizontal dist to interface: 43 km
# without subduction interface (UPF only)
lis_A1_dict = {'trace_center_x': cen1_x, 'trace_center_y': cen1_y, 'strike': 227.,
               'patch_heights': np.array([1.e3, 1.e3, 4.e3, 15.e3]),
               'patch_dips': np.array([80., 60., 38., 22.]),
               'geom_extension': "lis_A1"}

# with subduction interface
lis_A2_dict = {'trace_center_x': cen1_x, 'trace_center_y': cen1_y, 'strike': 227.,
               'patch_heights': np.array([1.e3, 1.e3, 4.e3, 15.e3, 3.e3]),
               'patch_dips': np.array([80., 60., 38., 22., 14.]),
               'geom_extension': "lis_A2"}

## model B: trace at Kidnappers ridge; ~same shape as Lachlan fault but 28 deg. at bottom
# depth to planar base: 6 km; horizontal dist to planar base: 5.87 km;
# depth to interface: 18 km; horizontal dist to interface: 28.5 km
# B1: without subduction interface (UPF only)
lis_B1_dict = {'trace_center_x': cen1_x, 'trace_center_y': cen1_y, 'strike': 227.,
               'patch_heights': np.array([1.e3, 1.e3, 4.e3, 12.e3]),
               'patch_dips':np.array([80., 60., 38., 28.]),
               'geom_extension': "lis_B1"}
lis_B2_dict = {'trace_center_x': cen1_x, 'trace_center_y': cen1_y, 'strike': 227.,
               'patch_heights': np.array([1.e3, 1.e3, 4.e3, 12.e3, 3.e3]),
               'patch_dips':np.array([80., 60., 38., 28., 13.]),
               'geom_extension': "lis_B1"}

## model C: trace at Kidnappers ridge; ~same shape as Lachlan fault but 25 deg. at bottom
# depth to planar base: 6 km; horizontal dist to planar base: 5.87 km;
# depth to interface:  km; horizontal dist to interface:  km
# C1: without subduction interface (UPF only)
lis_C1_dict = {'trace_center_x': cen1_x, 'trace_center_y': cen1_y, 'strike': 227.,
               'patch_heights': np.array([1.e3, 1.e3, 4.e3, 13.5e3]),
               'patch_dips':np.array([80., 60., 38., 25.]),
               'geom_extension': "lis_C1"}

# with subduction interface
lis_C2_dict = {'trace_center_x': cen1_x, 'trace_center_y': cen1_y, 'strike': 227.,
               'patch_heights': np.array([1.e3, 1.e3, 4.e3, 12.e3, 3.e3]),
               'patch_dips': np.array([80., 60., 38., 28., 13.]),
               'geom_extension': "lis_C2"}

## model F: trace at SE kidnappers basin; backthrust
# depth to fault base: 8.3 km
# without subduction interface (UPF only)
lis_F1_dict = {'trace_center_x': cen3_x, 'trace_center_y': cen3_y, 'strike': 47.,
               'patch_heights': np.array([1.e3, 1.e3, 3.e3, 3.e3]),
               'patch_dips':np.array([80., 60., 38., 22.]),
               'geom_extension': "lis_F1"}


## model P: trace at Kidnappers Ridge; planar 40 degree dip
# fault needs to go to 16.7 km depth to reach interface with 40 dip
# without subduction interface
plnr_P1_dict = {'trace_center_x': cen1_x, 'trace_center_y': cen1_y, 'strike': 227.,
                'patch_heights': np.array([8.e3, 8.7e3]),
                'patch_dips': np.array([40., 40.]),
                'geom_extension': "plnr_P1"}

# with 3 km (depth) subduction interface
plnr_P2_dict = {'trace_center_x': cen1_x, 'trace_center_y': cen1_y, 'strike': 227.,
                'patch_heights': ([16.7e3, 1.5e3, 1.5e3]),
                'patch_dips': np.array([40., 10.6, 12.1]),
                'geom_extension': "plnr_P2"}

# with 1 km (depth) subduction interface
plnr_P3_dict = {'trace_center_x': cen1_x, 'trace_center_y': cen1_y, 'strike': 227.,
                'patch_heights': ([16.7e3, 1.e3]),
                'patch_dips': np.array([40., 10.6]),
                'geom_extension': "plnr_P3"}

fault_geometries = [lis_A1_dict, lis_A2_dict, lis_B1_dict, lis_B2_dict, lis_C1_dict, lis_C2_dict, lis_F1_dict,
                                                plnr_P1_dict, plnr_P2_dict, plnr_P3_dict]

if not os.path.exists("out_data"):
    os.mkdir("out_data")

pkl.dump(fault_geometries, open(f"out_data/fault_geometries.pkl", "wb"))
