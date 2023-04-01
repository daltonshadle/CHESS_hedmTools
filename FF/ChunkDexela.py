#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:47:17 2022

@author: djs522
"""

import copy
import numpy as np
import yaml

from hexrd import instrument
from hexrd import config
from hexrd.imageseries import process
from hexrd.imageseries import save

# instrument
def load_instrument(yml):
    """
    Instantiate an instrument from YAML spec.

    Parameters
    ----------
    yml : str
        filename for the instrument configuration in YAML format.

    Returns
    -------
    hexrd.instrument.HEDMInstrument
        Instrument instance.

    """
    with open(yml, 'r') as f:
        icfg = yaml.safe_load(f)
    return instrument.HEDMInstrument(instrument_config=icfg)


def chunk_frame_cache(cfg_name, base_dim=(1944, 1536), n_rows_cols=(2, 2), 
                      row_col_gap=(0, 0), updated_config_path=None,
                      updated_instr_path=None, updated_analysis_name=None):
    '''
    

    Parameters
    ----------
    Parameters
    ----------
    cfg_name : string
        path to configuration file.
    base_dim : tuple (2 x 1), optional
        base dimensions for chunked panel. The default is (1944, 1536).
    n_rows_cols : tuple (2 x 1), optional
        number of chunked rows and cols of a panel. The default is (2, 2).
    row_col_gap : tuple (2 x 1), optional
        number of pixels to gap between each chunked panel. The default is (0, 0).
    updated_instr_path : string, optional
        path to new mpanel instr file to update in config. The default is None.

    Returns
    -------
    None. Saves chunked images to same path as original images. Saves updated config.

    '''    
    
    '''
    'image_series': {'format': 'frame-cache',
     'data': [{'file': '/home/djs522/additional_sw/hexrd3/example/examples/djs_mruby/mruby-1104-1_7_ff1_000011-cachefile.npz',
       'args': {},
       'panel': 'ff1'},
      {'file': '/home/djs522/additional_sw/hexrd3/example/examples/djs_mruby/mruby-1104-1_7_ff2_000011-cachefile.npz',
       'args': {},
       'panel': 'ff2'}]},
    '''
    
    # load configuration info
    cfg = config.open(cfg_name)[0]
    instr = load_instrument(cfg.instrument._configuration)
    img_dict = cfg.image_series
    new_cfg_img_data = []
    
    # process inputs arguments
    row_gap = row_col_gap[0]
    col_gap = row_col_gap[1]
    
    nrows = n_rows_cols[0]
    ncols = n_rows_cols[1]
    
    row_starts = [i*(base_dim[0] + row_gap) for i in range(nrows)]
    col_starts = [i*(base_dim[1] + col_gap) for i in range(ncols)]
    rr, cc = np.meshgrid(row_starts, col_starts, indexing='ij')
    
    # process chunking images
    for panel_id, panel in instr.detectors.items():
        image_base_name = img_dict[panel_id]._adapter._adapter._fname.replace(panel_id, '%s')
        for i in range(nrows):
            for j in range(ncols):
                # get new panel name
                panel_name = '%s_%d_%d' % (panel_id, i, j)
                
                # get new panel row and col chunk indexing
                rstr = rr[i, j]
                rstp = rr[i, j] + base_dim[0]
                cstr = cc[i, j]
                cstp = cc[i, j] + base_dim[1]
                
                # create chunked image and save
                r = [[rstr, rstp], [cstr, cstp]]
                ops = [('rectangle', r)]
                pims = copy.deepcopy(img_dict[panel_id])
                pims = process.ProcessedImageSeries(pims, ops)
                metad = pims.metadata
                metad['omega'] = img_dict[panel_id].__dict__['_omega']
                metad['panel_id'] = panel_name
                cache = image_base_name %(panel_name)
                save.write(pims, "dummy", fmt='frame-cache',
                                  style="npz",
                                  threshold=0, # not sure if the threshold should be 0 here...
                                  cache_file=cache)
                
                # update config image series list
                new_cfg_img_data.append({'file':cache, 'args':{}, 'panel':panel_name})
                
                pass
            pass
    
    # update config and save
    cfg._cfg['image_series']['data'] = new_cfg_img_data
    
    if updated_config_path is None:
        updated_config_path = cfg_name.replace('.yml', '_mpanel.yml')
    if updated_analysis_name is not None:
        cfg.analysis_name = updated_analysis_name
    if updated_instr_path is not None:
        cfg._cfg['instrument'] = updated_instr_path
    cfg.dump(updated_config_path)

def chunk_detector(cfg_name, base_dim=(1944, 1536), n_rows_cols=(2, 2), 
                      row_col_gap=(0, 0), updated_instr_path=None):
    '''
    

    Parameters
    ----------
    Parameters
    ----------
    cfg_name : string
        path to configuration file.
    base_dim : tuple (2 x 1), optional
        base dimensions for chunked panel. The default is (1944, 1536).
    n_rows_cols : tuple (2 x 1), optional
        number of chunked rows and cols of a panel. The default is (2, 2).
    row_col_gap : tuple (2 x 1), optional
        number of pixels to gap between each chunked panel. The default is (0, 0).
    updated_instr_path : string, optional
        path to new mpanel instr file to update in config. The default is None.

    Returns
    -------
    None. Saves chunked images to same path as original images. Saves updated config.

    '''    
    
    cfg = config.open(cfg_name)[0]
    instr = load_instrument(cfg.instrument._configuration)
    
    # process inputs arguments
    row_gap = row_col_gap[0]
    col_gap = row_col_gap[1]
    
    nrows = n_rows_cols[0]
    ncols = n_rows_cols[1]
    
    row_starts = [i*(base_dim[0] + row_gap) for i in range(nrows)]
    col_starts = [i*(base_dim[1] + col_gap) for i in range(ncols)]
    rr, cc = np.meshgrid(row_starts, col_starts, indexing='ij')
    
    icfg_dict = instr.write_config()
    new_icfg_dict = {}
    for key in icfg_dict.keys():
        new_icfg_dict[key] = icfg_dict[key]
    new_icfg_dict['detectors'] = {}
    
    if updated_instr_path is None:
        updated_instr_path = cfg.instrument._configuration.split('.')[0] + '_mpanel.yml'

    for panel_id, panel in instr.detectors.items():
        pcfg_dict = panel.config_dict(instr.chi, instr.tvec)['detector']
        for i in range(nrows):
            for j in range(ncols):
                panel_name = '%s_%d_%d' % (panel_id, i, j)

                rstr = rr[i, j]
                rstp = rr[i, j] + base_dim[0]
                cstr = cc[i, j]
                cstp = cc[i, j] + base_dim[1]
                
                ic_pix = 0.5*(rstr + rstp)
                jc_pix = 0.5*(cstr + cstp)

                sp_tvec = np.concatenate(
                    [panel.pixelToCart(np.atleast_2d([ic_pix, jc_pix])).flatten(),
                     np.zeros(1)]
                )

                tvec = np.dot(panel.rmat, sp_tvec) + panel.tvec

                # new config dict
                tmp_cfg = copy.deepcopy(pcfg_dict)

                # fix sizes
                tmp_cfg['pixels']['rows'] = base_dim[0]
                tmp_cfg['pixels']['columns'] = base_dim[1]

                # update tvec
                tmp_cfg['transform']['translation'] = tvec.tolist()

                new_icfg_dict['detectors'][panel_name] = copy.deepcopy(tmp_cfg)
                pass
            pass

    # write instrument
    with open(updated_instr_path, 'w') as fid:
        print(yaml.dump(new_icfg_dict, default_flow_style=False), file=fid)
    


if __name__ == '__main__':
    base_path = '/home/djs522/additional_sw/hedmTools/CHESS_hedmTools/Analysis/example_mruby/'
    
    # chunk detector
    cfg_name = base_path + 'mruby_calib_nov_2020_config.yml'
    chunk_detector(cfg_name, base_dim=(1944, 1536), n_rows_cols=(2, 2), 
                          row_col_gap=(0, 0), updated_instr_path=None)
    
    # chunk frame cache
    instr_name = base_path + 'dexela_nov_2020_instr_mpanel.yml'
    chunk_frame_cache(cfg_name, base_dim=(1944, 1536), n_rows_cols=(2, 2), 
                          row_col_gap=(0, 0), updated_instr_path=instr_name)
    