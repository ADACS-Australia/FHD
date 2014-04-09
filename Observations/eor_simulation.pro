PRO eor_simulation,cleanup=cleanup,recalculate_all=recalculate_all,export_images=export_images,version=version,$
    beam_recalculate=beam_recalculate,healpix_recalculate=healpix_recalculate,$
    channel=channel,output_directory=output_directory,save_visibilities=save_visibilities,$
    julian_day=julian_day,uvfits_version=uvfits_version,uvfits_subversion=uvfits_subversion,$
    silent=silent,combine_healpix=combine_healpix,start_fi=start_fi,end_fi=end_fi,skip_fi=skip_fi,$
    snapshot_healpix_export=snapshot_healpix_export,n_avg=n_avg,ps_kbinsize=ps_kbinsize,ps_kspan=ps_kspan,_Extra=extra
  except=!except
  !except=0
  heap_gc
  
  IF N_Elements(recalculate_all) EQ 0 THEN recalculate_all=0
  IF N_Elements(export_images) EQ 0 THEN export_images=1
  IF N_Elements(cleanup) EQ 0 THEN cleanup=0
  IF N_Elements(combine_healpix) EQ 0 THEN combine_healpix=0
  IF N_Elements(version) EQ 0 THEN version=1
  IF N_Elements(julian_day) EQ 0 THEN julian_day=2456528
  IF N_Elements(uvfits_version) EQ 0 THEN uvfits_version=2
  IF N_Elements(uvfits_subversion) EQ 0 THEN uvfits_subversion=0
  IF N_Elements(silent) EQ 0 THEN silent=0
  IF N_Elements(save_visibilities) EQ 0 THEN save_visibilities=1
  IF N_Elements(snapshot_healpix_export) EQ 0 THEN snapshot_healpix_export=1
  IF N_Elements(n_avg) EQ 0 THEN n_avg=2
  IF N_Elements(ps_kbinsize) EQ 0 THEN ps_kbinsize=3.
  IF N_Elements(ps_kspan) EQ 0 THEN ps_kspan=600.
  
  if n_elements(output_directory) eq 0 then output_directory='/nfs/mwa-09/r1/djc/EoR2013/Aug23/'
  image_filter_fn='filter_uv_uniform' ;applied ONLY to output images
  ;image_filter_fn=''
  
  ; This file structure works at MIT
  data_directory='/nfs/mwa-09/r1/EoRuvfits/jd'+strtrim(julian_day,2)+'v'+strtrim(uvfits_version,2)+'_'+strtrim(uvfits_subversion,2)
  
  
  vis_file_list=file_search(data_directory,'*.uvfits',count=n_files)
  fhd_file_list=fhd_path_setup(vis_file_list,version=version,output_directory=output_directory,_Extra=extra)
  healpix_path=fhd_path_setup(output_dir=output_directory,subdir='Healpix',output_filename='Combined_obs',version=version,_Extra=extra)
  catalog_file_path=filepath('MRC_full_radio_catalog.fits',root=rootdir('FHD'),subdir='catalog_data')
  ;calibration_catalog_file_path=filepath('mwa_calibration_source_list.sav',root=rootdir('FHD'),subdir='catalog_data')
  ;calibration_catalog_file_path=filepath('eor1_calibration_source_list.sav',root=rootdir('FHD'),subdir='catalog_data')
  ;calibration_catalog_file_path=filepath('mwa_calibration_source_list_nofornax.sav',root=rootdir('FHD'),subdir='catalog_data')
  ;calibration_catalog_file_path=filepath('eor01_calibration_source_list.sav',root=rootdir('FHD'),subdir='catalog_data')
  calibration_catalog_file_path=filepath('mwa_commissioning_source_list.sav',root=rootdir('FHD'),subdir='catalog_data')
  ;calibration_catalog_file_path=filepath('test_component_catalog.sav',root=rootdir('FHD'),subdir='catalog_data')
  
  dimension=2048.
  max_sources=20000.
  pad_uv_image=2.
  FoV=80.
  no_ps=1 ;don't save postscript copy of images
  min_baseline=1.
  min_cal_baseline=50.
  ring_radius=10.*pad_uv_image
  nfreq_avg=16
  ;max_calibration_sources=10000.
  no_rephase=1 ;set to use obsra, obsdec for phase center even if phasera, phasedec present in a .metafits file
  ;no_fits=1
  combine_obs=0
  smooth_width=11.
  bandpass_calibrate=1
  calibration_polyfit=2
  no_restrict_cal_sources=1
  
  ;; for simulation
  eor_sim = 1
  include_catalog_sources = 0
  
  ;; stuff from general_obs not in eor_firstpass
  ;Set up paths
  ;Convoluted way of setting up 'instrument' for use here, while still leaving it to be passed in Extra
  IF N_Elements(extra) GT 0 THEN IF Tag_exist(extra,'instrument') THEN instrument=extra.instrument
  IF N_Elements(instrument) EQ 0 THEN instrument='mwa'
  
  IF N_Elements(double_precison_beam) EQ 0 THEN double_precison_beam=0
  ;Set up gridding and deconvolution parameters
  IF N_Elements(complex_beam) EQ 0 THEN complex_beam=1
  IF N_Elements(n_pol) EQ 0 THEN n_pol=2
  IF N_Elements(precess) EQ 0 THEN precess=0 ;set to 1 ONLY for X16 PXX scans (i.e. Drift_X16.pro)
  
  IF N_Elements(start_fi) EQ 0 THEN start_fi=0
  fi=start_fi
  IF N_Elements(end_fi) GT 0 THEN n_files=end_fi+1 ;changed to allow end_fi and update to both be specified
  WHILE fi LT n_files DO BEGIN
    IF ~Keyword_Set(silent) THEN print,String(format='("On observation ",A," of ",A)',Strn(Floor(fi-start_fi+1)),Strn(Floor(n_files-start_fi)))
    IF N_Elements(skip_fi) GT 0 THEN BEGIN
      IF max(skip_fi EQ fi) GT 0 THEN BEGIN
        fi+=1
        CONTINUE
      ENDIF
    ENDIF
    ;    IF (recalculate_all EQ 0) AND Keyword_Set(cleanup) THEN BEGIN IF N_Elements(fi_use) GT 0 THEN fi_use=[fi_use,fi] ELSE fi_use=fi & fi+=1 & CONTINUE & ENDIF
    ;    IF Keyword_Set(force_no_data) THEN BEGIN IF N_Elements(fi_use) GT 0 THEN fi_use=[fi_use,fi] ELSE fi_use=fi & fi+=1 & CONTINUE & ENDIF
    fhd_sim,vis_file_list[fi],file_path_fhd=fhd_file_list[fi],n_pol=n_pol,recalculate_all=recalculate_all,$
      beam_recalculate=beam_recalculate, /silent,max_sources=max_sources, $
      eor_sim=eor_sim, include_catalog_sources = include_catalog_sources, catalog_file_path=catalog_file_path,source_list=source_list, $
      model_uvf_cube=model_uvf_cube, model_image_cube=model_image_cube,$
      export_images=export_images,dimension=dimension,image_filter_fn=image_filter_fn,pad_uv_image=pad_uv_image,$
      complex=complex_beam,double=double_precison_beam,precess=precess,error=error,weights_grid=weights_grid,$
      save_visibilities=save_visibilities,healpix_recalculate=healpix_recalculate,$
      snapshot_healpix_export=snapshot_healpix_export,_Extra=extra
      
    IF Keyword_Set(error) THEN BEGIN
      print,'###########################################################################'
      print,'###########################################################################'
      print,'###########################################################################'
      print,'Error encountered!'
      print,'###########################################################################'
      print,'###########################################################################'
      print,'###########################################################################'
    ENDIF ELSE $
      IF N_Elements(fi_use) GT 0 THEN fi_use=[fi_use,fi] ELSE fi_use=fi
    fi+=1.
  ENDWHILE
  IF N_Elements(end_fi) EQ 0 THEN end_fi=fi-1
  
  n_files_use=N_Elements(fi_use)
  vis_file_list=vis_file_list[fi_use]
  fhd_file_list=fhd_file_list[fi_use]
  
  IF Keyword_Set(cleanup) THEN FOR fi=0L,n_files_use-1 DO fhd_cleanup,fhd_file_list[fi],_Extra=extra
  
  heap_gc
  
  !except=except
END