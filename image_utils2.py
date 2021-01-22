
def list_bands(image_path, bands):
    band_list = list()
    listdir = os.listdir(image_path)
    for filename in listdir:
        if bands in filename:
            band_list.append(str(image_path) + '/' + str(filename))
    band_list.sort()
    return band_list


def arosics_local(project_dir, reference_list, target_list):
    from arosics import COREG_LOCAL

    kwargs = {
        'grid_res'     : 200,
        'window_size'  : (64,64),
        'path_out'     : 'auto',
        'projectDir'   : project_dir,
        'q'            : False,
    }

    for reference, target in zip(reference_list, target_list):
        print(reference)
        print(target)
        CRL = COREG_LOCAL(reference, target, **kwargs,fmt_out='GTIFF')
        CRL.correct_shifts()
        #CRL.view_CoRegPoints(figsize=(15,15), backgroundIm='tgt')


def aeronet_prepare(filename):
    dateparse = lambda x: datetime.strptime(x, "%d:%m:%Y %H:%M:%S")
    pd_name = pd.read_csv(filename, skiprows=6, na_values=['-999'], parse_dates={'times':[0,1]}, date_parser=dateparse)

    # Drop any row or column fully of NaNs and sort by the index
    pd_name = (pd_name.dropna(axis=1, how='all')
            .dropna(axis=0, how='all')
            .rename(columns={'Last_Processing_Date(dd/mm/yyyy)': 'Last_Processing_Date'})
            .sort_index())
    
    # Split times in new variable - day + hour
    pd_name["times"] = pd_name["times"].astype(str)
    pd_name["day"]= pd_name["times"].str.split(" ", n = 1, expand = True)[0].str.replace('-', '')
    pd_name["hour"]= pd_name["times"].str.split(" ", n = 1, expand = True)[1] 

    # Filter to satellite observation using hour
    sat_pass_filter =  pd_name['hour']<'11:00:00'
    af_sat_pass_filter = pd_name[sat_pass_filter]
    sat_pass_filter =  af_sat_pass_filter['hour']>'10:00:00'
    af_sat_pass_filter = af_sat_pass_filter[sat_pass_filter]

    #drop duplicate daily filtered aod values
    af_sat_pass_filter = af_sat_pass_filter.drop_duplicates(subset=['day'])
    
    print(af_sat_pass_filter.shape)

    return af_sat_pass_filter


def list_bands(image_pattern, bands):
    # Creates an empty list
    band_list = list()
    for band in bands:
        img_path = image_pattern.format(band)
        band_list.append(img_path)
    return band_list


def list_bands_match(image_path, bands):
    matching = [s for s in os.listdir(image_path) if any(xs in s for xs in bands)]
    matching.sort()
    return matching


def list_bands_mod(image_path, bands):
    band_list = list()
    listdir = os.listdir(image_path)
    for band in bands:
        for filename in listdir:
            if band in filename:
                band_list.append(filename)
    return band_list


def get_imgs_by_band(image_path, band):
    '''This function works with an dir path and a band (.tif) to be returned in that folders and subfolders
    retrieve a list of specific bands.
    img_list retrieve strings band paths
    band = band string that user need retrive. i.e.: red, nir, blue, B02, B01, etc.'''
    ###Create the empty list
    img_list = list()
    ###for in the folders to retrieve specific band in path, subdirs and files - using os.walk
    for path, subdirs, files in os.walk(image_path):
        for name in files:
            if name.endswith(band + '.tif'):
                img_list.append(os.path.join(image_path,name))
    return img_list

def plot_doxani(hist_data, hist_min, hist_max, lines, line_names, bins, colors=['tab:pink', 'tab:red', 'tab:green', 'tab:purple'], marker=['', '+', 'x', '*'], xlabel=None, global_text=None, out_file=None):
    # Plot Doxane
    fig, ax = plt.subplots()
    if xlabel:
        ax.set_xlabel(xlabel)

    # Plot lines
    i=0
    for line in lines:
        ax.plot(bins, line, color=colors[i], label=line_names[i], marker=marker[i])
        i+=1

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax.twinx()

    # Order of each plot (Front, Back)
    ax.set_zorder(2)
    ax2.set_zorder(1)
    ax.patch.set_visible(False)

    # Remove scientific notation
    ax2.ticklabel_format(style='plain')

    # the histogram of the data
    _, _, _ = ax2.hist(hist_data, bins, edgecolor="k", label = 'nb of points')

    if global_text is not None:
        xpos = bins[0]
        # Get text y pos
        ypos = 0
        for line in lines:
            if numpy.nanmax(line) > ypos:
                ypos = numpy.nanmax(line)*0.25
        box_text = f'APU {global_text[0]} Band {global_text[1]} \n{global_text[2]}\nnbp={global_text[3]}\nAvg Truth {global_text[4]:.2f}\nAccuracy{global_text[5]:.2f}\nPrecision{global_text[6]:.2f}\nUncertainty{global_text[7]:.2f}'
        ax.text(x = xpos, y = ypos, s = box_text)

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()

      
def linregress_band(img1, img2, output_folder, out_file=None):
    
    #os.chdir(dir) vai para o diretório definido em os.chdir
    os.chdir(output_folder)
    
    x_label=None
    y_label=None
    slope, intercept, r_value, p_value, std_err = linregress(img1, img2)
    r2 = r_value**2

    print("slope:{}".format(slope))
    print("intercept:{}".format(intercept))
    print("r_value:{}".format(r_value))
    print("p_value:{}".format(p_value))
    print("std_error:{}".format(std_err))
    print("r-squared:{}".format(r2))

    #plota histograma 2d
    fig = plt.figure(figsize=(10, 10), facecolor='w')
    ax1 = fig.add_subplot(111)
    textstr = " n={}\n R = {:.4f} \n {} = {:.4f} \n stderr = {:.4f} \n intercept={:.4f}\n slope={:.4f}".format(img1.shape[0], r_value, '${R^2}$', r2, std_err, intercept, slope)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.01)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    cmin=0
    cmax=10000
    n_bins=100
    plt.hist2d(img1, img2, bins=(n_bins, n_bins), cmin=5, range=((cmin, cmax), (cmin, cmax)), cmap='plasma')

    # # desenho da reta, dados 2 pontos extremos
    x2 = numpy.array([0, 10000])
    plt.plot(x2, x2, color = ('#808080'), ls='dashed', linewidth=1)
    plt.plot(x2, slope * x2 + intercept, '--k', linewidth=1)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xlabel('4')
    plt.ylabel("4A")
    plt.title(out_file)
    plt.colorbar()
    plt.savefig(out_file +'.png', dpi=300, bbox_inches='tight')
    plt.close(fig=None)


def gdal_repro(input_file, EPSG, output_file):
    ### Using gdal warp to reproject stack tif

    # Set input file name to warp
    filename = input_file

    # Open the input file with gdal.Open
    input_raster = gdal.Open(filename)

    # Set output file name
    output_raster = output_file

    # Warp the input tif to selected EPSG in dstSRS
    gdal.Warp(output_raster,input_raster,dstSRS=EPSG)

    # Clean memory
    del input_raster
    del output_raster


def list_all_images(image_path):
    # Creates an empty list
    li_bands = []
    for name in os.listdir(image_path):
            li_bands.append(os.path.join(image_path,name))
    li_bands.sort()
    return li_bands


def get_crops(path_out, inputshape, inputraster):
    #this is considered by the processing time of stmetrics
    #path_out = '/home/fronza/Downloads/ST_METRICS/cb_4'
    #set working output dir
    os.chdir(path_out)

    #open shapefile 
    with fiona.open(inputshape, "r") as shapefile:
        for feature in shapefile:
            # obtendo a geometria associada a feição
            geom = shape(feature["geometry"])
            #print(feature)
            #open tif to crop by selected ids
            with rasterio.open(inputraster, "r") as src:
                out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True)
                out_meta = src.meta
            #print(str(feature['properties']['Id']))
            #output file name 
            out_file = 'crop_' + str(feature['properties']['Id']) + '_masked.tif'
            #metadata write
            out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
            #write output in crop tif 
            with rasterio.open(out_file, "w", **out_meta) as dest:
                dest.write(out_image)
        listcrop = list_all_images(path_out)
        listcrop.sort()
        return listcrop


def stmetrics_crops(input_list):
    #initialize empty list variable
    im_crop = []
    
    #loop to run sits2metrics for cropped images
    for crop in input_list: 
        dataset = rasterio.open(crop)
        #metricas = numpy.vstack(lista)
        im_crop.append(stmetrics.metrics.sits2metrics(dataset))
    return im_crop


def plot_crops(crops_arrays):
    #create header to metrics in stmetrics lib # 27 metrics update
    header = ['mean_ts',  'max_ts',  'min_ts', 'std_ts',  'sum_ts',  'amplitude_ts', 'fslope_ts',  'skew_ts',
     'amd_ts',  'abs_sum_ts', 'fqr_ts',  'sqr_ts',  'tqr_ts',  'iqr_ts',  'area_ts', 'area_q1',  'area_q2',  
              'area_q3',  'area_q4',  'ecc_metric',  'gyration_radius',  'polar_balance',  'angle',  
              'dfa_fd',  'hurst_exp',  'katz_fd']


    #initialize figure (all) and ax with 19 rows and each crop in new column
    fig, ax  = plt.subplots(26, len(crops_arrays) , figsize=(10,30))
    
    #loop to create columns on lenght of cropped tifs
    for column in range(1,len(crops_arrays)+1):
        #print(column)
        for b, n in zip(range(1, im_crop[0].shape[0]+1),header):
            #b is the metrics index - over 19 metrics
            #n is the header index  - over 19 metrics names
            #column to control the column ax -column index
            #print(column, b, n)
            # here we walk through ax row and column to plot with imshow
            ax[b-1, column-1].imshow(crops_arrays[column-1][b-1,:,:])
            plt.tight_layout()
            #set header title to ax
            ax[b-1, column-1].set_title(n)
        #show figure
        #plt.show()
        #extras - to save figure
    #fig.suptitle('Título ', fontsize=16)
    fig.savefig('out_crop' +'.png', dpi=300, bbox_inches='tight')
    plt.close(fig=None)


def crop_metrics_tocsv(input_list, stmetrics_result, output_file):
    header = ['mean_ts',  'max_ts',  'min_ts', 'std_ts',  'sum_ts',  'amplitude_ts', 'fslope_ts',  'skew_ts',
     'amd_ts',  'abs_sum_ts', 'fqr_ts',  'sqr_ts',  'tqr_ts',  'iqr_ts',  'area_ts', 'area_q1',  'area_q2',  
              'area_q3',  'area_q4',  'ecc_metric',  'gyration_radius',  'polar_balance',  'angle',  
              'dfa_fd',  'hurst_exp',  'katz_fd']
    l = []
    for group, i  in zip(range(0, len(stmetrics_result)), range(0, len(stmetrics_result))):
        l_test = []
        #print(group)
        l_test.append(input_list[i])
        l.append(l_test)
        for item, h in zip(stmetrics_result[group], range(0, len(header))):
            metrics = item.mean()
            #print(metrics)
            l_test.append(header[h])
            l_test.append(metrics)
    with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(l)


def median_crops(crops_list):
    # initialize empty list variable
    l = []

    #for to get mean value to crop in 36 time points
    for crop in crops_list:
        #initialize and clear totalcrop for an 36 time poins stack
        totalcrop = []
        #append totalcrop in a list of lists called l
        l.append(totalcrop)
        #open crop
        with rasterio.open(crop) as cropimg:
            #iterate over bands
            for b in range(1, cropimg.count+1):
                #read band
                data = cropimg.read(b)
                #calculate mean not considering no data -9999 values
                average = data[data!=-9999].mean()
                #print(b, average)
                #append the mean in totalcrop list
                totalcrop.append(average) 
                #print(b, mean)
                #create a variable num_time
                num_time = cropimg.count
                #print(num_time)
    return l


def serie_temporal_plot(median_crops_calculated, list_crops):
    #create a time range to use in plot
    num_time = len(median_crops_calculated[0])
    time = range(1, num_time+1)

    #valores_cro1 = l[0]
    lencrop = (len(median_crops_calculated))

    #initialize figure with crop lenght in rows
    fig, ax  = plt.subplots(nrows=lencrop,ncols=1, figsize=(10,180))

    #loop to create rows in lenght of cropped tifs
    for b in range(0, len(median_crops_calculated)):
        #save crop in valores crop to use in plot
        valores_crop = median_crops_calculated[b]
        #ax plot the time and valores_crop
        ax[b].plot(time,valores_crop)
        #plt.tight_layout()
        #set header title to ax
        n = os.path.basename(list_crops[b])[0:-4]
        ax[b].set_title(n)
    #show figure
    #plt.tight_layout()
    plt.show()
    #fig.savefig('out_series' +'.png', dpi=500, bbox_inches='tight')


def crop_get_proj(input_list, shapefile):
    with rasterio.open(input_list[0]) as raster_crs:
        crop_raster_profile = raster_crs.profile
        crop_bound_cube = shapefile.to_crs(crop_raster_profile["crs"])
    return crop_bound_cube


def mean_band(cropped_array):
    # initialize empty list variable
    meanband = []
    #for to get mean value to crop in time points
    for crop in cropped_array:
        average = crop[crop!=-9999].mean()
        average = average/10000
        meanband.append(average)
    return meanband


def output_s2(image_path, li_bands):
    # Set current working dir to SENTINEL-2X image DIR
    os.chdir(image_path)
    logging.info('{}'.format(li_bands))
    #Set Virtual Raster options
    vrt_options = gdal.BuildVRTOptions(separate='-separate')
    #Set output tif filename
    output_filename = image_path.split("/")[-0] + '_stk.tif'
    # Create virtual raster
    ds = gdal.BuildVRT('img{}.vrt'.format(output_filename), li_bands, options=vrt_options)
    #Create output tif
    gdal.Translate(output_filename, ds, format='GTiff')


def output_lc8(image_path, li_bands):
    # Set current working dir to LANDSAT-8 image DIR
    os.chdir(image_path)
    logging.info('{}'.format(li_bands))
    #Set Virtual Raster options
    vrt_options = gdal.BuildVRTOptions(separate='-separate')
    #Set output tif filename
    output_filename = image_path.split("/")[-0] + '_stk.tif'
    # Create virtual raster
    ds = gdal.BuildVRT('img{}.vrt'.format(output_filename), li_bands, options=vrt_options)
    #Create output tif
    gdal.Translate(output_filename, ds, format='GTiff')


def list_bands(image_path, patterns):
    # Creates an empty list
    li_bands = []
    # Search Sentinel-2 blue, green, red, nir and insert to the list
    for filename in os.listdir(image_path):
        if filename.endswith((patterns)):
            li_bands.append(filename)
    li_bands.sort()
    return li_bands


def stack_virtual_raster(image_path, pattern):
    #seleciona qual pattern para construir o stk TODO aprimorar para o pacote
    li_bands1 = list_bands(image_path1, bands_s2_toa)
    [x for _, x in sorted(zip(pattern, li_bands1))]
    output_lc8(image_path1, li_bands1)

    li_bands2 = list_bands(image_path2, bands_lc8_toa)
    [x for _, x in sorted(zip(pattern, li_bands2))]
    output_s2(image_path2, li_bands2)


def apu_calc(data1, data2):
    '''This function compute APU metrics.
    :param data1: Array of pixel values of a single band.
    :type data1: numpy.array
    :param data2: Array of pixel values of a single band.
    :type data2: numpy.array
    '''
    assert data1.size==data2.size
    sample_size = data1.size # Total pixels
    residuals = data2 - data1 # Residuals calc
    acc = residuals.sum()/sample_size # Accuracy metric
    diff_a = numpy.power((residuals - acc), 2)
    prec = numpy.sqrt(diff_a.sum()/(sample_size-1)) # Precision metric
    residuals_square = numpy.power((residuals), 2)
    unc = numpy.sqrt(residuals_square.sum()/sample_size) # Uncertainty
    avg_truth = data1.mean()
    return sample_size, acc, prec, unc, avg_truth


def list_pattern(image_path, pattern):
    band_list = list()
    listdir = os.listdir(image_path)
    for filename in listdir:
        if pattern in filename:
            band_list.append(str(image_path) + '/' + str(filename))
    band_list.sort()
    return band_list


def list_pattern_startswith(image_path, pattern):
    band_list = list()
    listdir = os.listdir(image_path)
    for filename in listdir:
        if filename.startswith(pattern):
            band_list.append(str(image_path) + '/' + str(filename))
    band_list.sort()
    return band_list


def plot_doxani(hist_data, hist_min, hist_max, lines, line_names, bins,
                colors=['tab:pink', 'tab:red', 'tab:green', 'tab:purple'], marker=['', '+', 'x', '*'],
                xlabel=None, global_text=None, out_file=None):
    # Plot Doxane
    fig, ax = plt.subplots()
    if xlabel:
        ax.set_xlabel(xlabel)

    # Plot lines
    i=0
    for line in lines:
        ax.plot(bins, line, color=colors[i], label=line_names[i], marker=marker[i])
        i+=1

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax.twinx()

    # Order of each plot (Front, Back)
    ax.set_zorder(2)
    ax2.set_zorder(1)
    ax.patch.set_visible(False)

    # Remove scientific notation
    ax2.ticklabel_format(style='plain')

    # the histogram of the data
    _, _, _ = ax2.hist(hist_data, bins, edgecolor="k", label = 'nb of points')

    if global_text is not None:
        xpos = bins[0]
        # Get text y pos
        ypos = 0
        for line in lines:
            if numpy.nanmax(line) > ypos:
                ypos = numpy.nanmax(line)*0.25
        box_text = f'APU {global_text[0]} Band {global_text[1]} \n{global_text[2]}\nnbp={global_text[3]}\nAvg Truth {global_text[4]:.2f}\nAccuracy{global_text[5]:.2f}\nPrecision{global_text[6]:.2f}\nUncertainty{global_text[7]:.2f}'
        ax.text(x = xpos, y = ypos, s = box_text)
    
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()


def sug_spec_on_reflec(reflec):
    return 0.005 + 0.05 * reflec


def parcelise_data(r1, r2, bins):
    # Calculate APU per parcel
    data_bins = numpy.digitize(r1, bins)

    accuracy = list()
    precision = list()
    uncertainty = list()
    for parcel in range(1, len(bins)+1):
        parc_pos = numpy.where(data_bins == parcel)[0]
        if parc_pos is not None and len(parc_pos)>200:
            _, acc, prec, unc, _ = apu_calc(r1[parc_pos], r2[parc_pos])
        else:
            _, acc, prec, unc, _ = numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan
        accuracy.append(acc)
        precision.append(prec)
        uncertainty.append(unc)
    return accuracy, precision, uncertainty


def metrics_and_plot(r1, r2, out_file=None, num_valid_observations=None):
    r1_min = numpy.nanmin(r1)
    r1_max = numpy.nanmax(r1)

    num_bins = int(numpy.ceil((r1_max - r1_min)*100))

    r1_floor = numpy.floor(r1_min*100)/100 #*100 to remove 2 decimal places and / 100 to return scale
    r1_ceil = numpy.ceil(r1_max*100)/100 #*100 to remove 2 decimal places and / 100 to return scale

    bins = numpy.linspace((r1_floor), (r1_ceil), num_bins, endpoint=False)

    # Remove from r1 and r2 observations with less than num_valid_observations on its bin
    if num_valid_observations is not None:
        data_bins = numpy.digitize(r1, bins)
        bincount = numpy.bincount(data_bins)
        valid_bins = numpy.where(bincount > num_valid_observations)

        r1_floor = bins[valid_bins[0][0]-1]
        r1_ceil = bins[valid_bins[0][-1]-1]+0.01
        num_bins = int((r1_ceil - r1_floor)*100) #End of the larger parcel - start of smallest parcel
        bins = numpy.linspace((r1_floor), (r1_ceil), num_bins, endpoint=False)

        # Check which positions from r1 do not have enough num_valid_observations
        values_to_keep = numpy.isin(data_bins, valid_bins[0])
        r1 = r1[values_to_keep]
        r2 = r2[values_to_keep]

    sug_spec = numpy.array(list(map(sug_spec_on_reflec, bins)))
    sample_size, acc, prec, unc, avg_truth = apu_calc(r1, r2) # Global APU
    acc_parc, prec_parc, unc_parc = parcelise_data(r1, r2, bins) # Per parcel APU
    
    acc_parc = numpy.abs(acc_parc)
    prec_parc = numpy.abs(prec_parc)
    unc_parc = numpy.abs(unc_parc)   
    
    lines = [sug_spec, acc_parc, prec_parc, unc_parc]
    
    
    line_names = ['suggested specs', 'accuracy', 'precision', 'uncertainty']

    global_text = ['CBERS 4 MUX', 'BAND', 'Dataset CBERS 4 MUX', sample_size, avg_truth, acc, prec, unc]

    plot_doxani(hist_data=r1, hist_min=r1_floor, hist_max=r1_ceil, lines=lines,
                line_names=line_names, bins=bins, 
                xlabel='CBERS 4 CMPAC', global_text=global_text, out_file=out_file)


def linregress_band(img1, img2, output_folder, max_scale=None, out_file=None):


    
    #os.chdir(dir) vai para o diretório definido em os.chdir
    os.chdir(output_folder)
    
    x_label=None
    y_label=None
    slope, intercept, r_value, p_value, std_err = linregress(img1, img2)
    r2 = r_value**2

    print("slope:{}".format(slope))
    print("intercept:{}".format(intercept))
    print("r_value:{}".format(r_value))
    print("p_value:{}".format(p_value))
    print("std_error:{}".format(std_err))
    print("r-squared:{}".format(r2))

    #plota histograma 2d
    fig = plt.figure(figsize=(10, 10), facecolor='w')
    ax1 = fig.add_subplot(111)
    textstr = " n={}\n R = {:.4f} \n {} = {:.4f} \n stderr = {:.4f} \n intercept={:.4f}\n slope={:.4f}".format(img1.shape[0], r_value, '${R^2}$', r2, std_err, intercept, slope)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.01)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    cmin=0
    cmax=max_scale
    n_bins=100
    plt.hist2d(img1, img2, bins=(n_bins, n_bins), cmin=5, range=((cmin, cmax), (cmin, cmax)), cmap='plasma')

    # # desenho da reta, dados 2 pontos extremos
    x2 = numpy.array([0, max_scale])
    plt.plot(x2, x2, color = ('#808080'), ls='dashed', linewidth=1)
    plt.plot(x2, slope * x2 + intercept, '--k', linewidth=1)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.xlabel('CMPAC')
    plt.ylabel("MS3")
    plt.title(out_file)
    plt.colorbar()
    plt.savefig(out_file +'.png', dpi=300, bbox_inches='tight')
    plt.close(fig=None)


def diff_sum_abs(image_path1,image_path2, output_folder):
    '''
    documentar para sair automaticamente
    '''
    # get o stk de bandas .tif
    for filename in os.listdir(image_path1):
        if filename.endswith('.tif'):
            i1 = filename
            ds1 = gdal.Open(os.path.join(image_path1, filename))

    # Set current working dir to second image
    os.chdir(image_path2)

    for filename in os.listdir(image_path2):
        if filename.endswith('.tif'):
            i2 = filename
            ds2 = gdal.Open(os.path.join(image_path2, filename))

     # Create GTIF file
    driver = gdal.GetDriverByName("GTiff")

    #conta numero de bandas
    numbands = ds1.RasterCount
    print(numbands)

    #define nome do output
    # cria o nome do arquivo de saída
    output_file = os.path.basename(i1) + "_DIF_ABS_" + os.path.basename(i2)
    print(output_file)

    #ref t1 banda 1
    xsize = ds1.RasterXSize
    ysize = ds1.RasterYSize

    #cria a imagem de saída
    os.chdir(output_folder)
    dataset = driver.Create(output_file, xsize, ysize, 1, gdal.GDT_Float32)

    # follow code is adding GeoTranform and Projection
    geotrans=ds1.GetGeoTransform()  #get GeoTranform from existed 'data0'
    proj=ds1.GetProjection() #you can get from a exsited tif or import
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)

    #cria lista vazia para receber os dados
    results = []

    os.chdir(output_folder)
    for band in range(numbands):
        #ds1 = gdal.Open(os.path.join(img1, listbands1[band]))
        #ds2 = gdal.Open(os.path.join(img2, listbands2[band]))
        bandtar = ds1.GetRasterBand(band+1).ReadAsArray()
        bandref = ds2.GetRasterBand(band+1).ReadAsArray()
        # transforma para float
        bandtar = bandtar.astype(float)
        bandref = bandref.astype(float)
        #bandtar = np.array(ds1.GetRasterBand(band).ReadAsArray().astype(float))
        #bandref = np.array(ds2.GetRasterBand(band).ReadAsArray().astype(float))
        results.append(np.abs(bandtar - bandref))
        diff_abs_sum = np.sum(results, axis=0)
        dataset.GetRasterBand(1).WriteArray(diff_abs_sum)


def diff_bands_stats(input_folder, output_folder):
    '''
    documentar para sair automaticamente
    '''
    #inicializa variavel

    #pega a imagem de diferença na pasta
    for filename in os.listdir(input_folder):
        if '_DIF_' in filename:
            diffname = filename
            diffimg = gdal.Open(os.path.join(input_folder, filename))

    #os.chdir(dir) vai para o diretório definido em os.chdir
    os.chdir(output_folder)

    #conta bandas
    bands = diffimg.RasterCount
    
    #Define label x e y
    x_label=None
    y_label=None
      
    #compute histogram to each band
        for b in range(1, bands+1):
        data = diffimg.GetRasterBand(b).ReadAsArray()
        dmin = np.min(data) #min value
        dmax = np.max(data) #max value
        mean = np.mean(data) #calculate mean 
        median = np.median(data) #calculate median without value 0
        std = np.std(data) #calculate std without value 0
        var = np.var(data) #calculate var without value 0
        histog = np.hstack(data)	#Compute the histogram of a set of data.
        #define histogram intervals
        a = [-2000, -1000, -500, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000]
        print("[ STATS DIFF] =  Band=%.1d, Min=%.3f, Max=%.3f, Mean=%.3f, Median=%.3f, StdDev=%.3f, Variance=%.3f" % ((b), dmin, dmax, mean, median, std, var))
        #create fig    
        fig = plt.figure(figsize=(10, 10), facecolor='w')
        #create subplot to histogram stats
        ax1 = fig.add_subplot(111)
        #plot text in subplot
        textstr = "Band={}\n Dmin={:.4f} \n Dmax={:.4f} \n Mean= {:.4f} \n Median={:.4f}\n Std={:.4f} \n Variance={:.4f} \n".format(b, dmin, dmax, mean, median,std,var)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.01)
        #subplot dimensions
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)
        #plot histogram
        plt.hist(histog, bins=a)
        #define label e fonte de label
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        #define label dos eixos x e y
        plt.xlabel("band_" + str(b))
        plt.ylabel("band_" + str(b))
        #plt.colorbar()
        print(os.path.basename(filename))
        out_file = os.path.basename(filename) + "_band_" + str(b)
        plt.savefig(out_file +'.png', dpi=300, bbox_inches='tight')
        plt.close(fig=None)


def compare_linregress(image_path1,image_path2, output_folder):
    #pega o stk de bandas .tif img1
    for filename in os.listdir(image_path1):
        if filename.endswith('.tif'):
            i1 = filename
            print(i1)
            ds1 = gdal.Open(os.path.join(image_path1, filename))

    #pega o stk de bandas .tif img2
    for filename in os.listdir(image_path2):
        if filename.endswith('.tif'):
            i2 = filename
            print(i2)
            ds2 = gdal.Open(os.path.join(image_path2, filename))

    #os.chdir(dir) vai para o diretório definido em os.chdir
    os.chdir(output_folder)

    #deve ser feito um loop pra iterar nas bandas
    numbands = ds1.RasterCount
    print(numbands)

    for b in range(numbands):
        bandref = np.array(ds1.GetRasterBand(b+1).ReadAsArray())
        bandtar = np.array(ds2.GetRasterBand(b+1).ReadAsArray())
        #Convert np array to float
        bandref = bandref.astype(float)
        bandtar = bandtar.astype(float)
        #set NaN == -9999
        bandref[bandref== -9999]=np.nan
        bandtar[bandtar== -9999]=np.nan
        # A 1-D array, containing the elements of the input, is returned. 
        x = bandref.ravel()
        y = bandtar.ravel()
        #mask in NaN data
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]
        # # desenho da reta, dados 2 pontos extremos
        x_label=None
        y_label=None
        out_file = os.path.basename(i1) + "__" + str(b+1)+ "__" + os.path.basename(i2)
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        r2 = r_value**2

        print("slope:{}".format(slope))
        print("intercept:{}".format(intercept))
        print("r_value:{}".format(r_value))
        print("p_value:{}".format(p_value))
        print("std_error:{}".format(std_err))
        print("r-squared:{}".format(r2))

        #plota histograma 2d
        fig = plt.figure(figsize=(10, 10), facecolor='w')
        ax1 = fig.add_subplot(111)
        textstr = " n={}\n R = {:.4f} \n {} = {:.4f} \n stderr = {:.4f} \n intercept={:.4f}\n slope={:.4f}".format(x.shape[0], r_value, '${R^2}$', r2, std_err, intercept, slope)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.01)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)
        cmin=0
        cmax=10000
        n_bins=500
        plt.hist2d(x, y, bins=(n_bins, n_bins), cmin=5, range=((cmin, cmax), (cmin, cmax)), cmap='plasma')

        # # desenho da reta, dados 2 pontos extremos
        x2 = np.array([0, 10000])
        plt.plot(x2, x2, color = ('#808080'), ls='dashed', linewidth=1)
        plt.plot(x2, slope * x2 + intercept, '--k', linewidth=1)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.xlabel(os.path.basename(i1) + "_band_"+ str(b+1))
        plt.ylabel(os.path.basename(i2) + "_band_"+ str(b+1))
        plt.colorbar()
        plt.savefig(out_file +'.png', dpi=300, bbox_inches='tight')
        plt.close(fig=None)


def diff_bands_stk(image_path1, image_path2, output_folder):
    #Set current working dir to first image
    os.chdir(image_path1)

    #get o stk de bandas .tif
    for filename in os.listdir(image_path1):
        if filename.endswith('_stk.tif'):
            i1 = filename
            ds1 = gdal.Open(os.path.join(image_path1, filename))
    #Set current working dir to second image
    os.chdir(image_path2)

    for filename in os.listdir(image_path2):
        if filename.endswith('_stk.tif'):
            i2 = filename
            ds2 = gdal.Open(os.path.join(image_path2, filename))

    #define o range de bandas da comparação
    numbands = ds1.RasterCount
    
    #define o driver
    driver = gdal.GetDriverByName("GTiff")

    #cria o nome do arquivo de saída
    output_file = os.path.basename(i1) + "_DIF_" + os.path.basename(i2)
    print(output_file)
    os.chdir(output_folder)
    print("CURRENTDIR:" + os.getcwd())

    #dimensoes do raster
    xsize = ds1.RasterXSize
    ysize = ds1.RasterYSize

    #cria o dataset vazio com as bandas da diferença
    dataset = driver.Create(output_file, xsize, ysize, numbands, gdal.GDT_Float32)

    # follow code is adding GeoTranform and Projection
    geotrans=ds1.GetGeoTransform()  #get GeoTranform from existed 'data0'
    proj=ds1.GetProjection() #you can get from a exsited tif or import 
    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)

    #setting nodata value
    dataset.GetRasterBand(1).SetNoDataValue(-9999)

    #loop da diferença para gerar as bandas
    for b in range(numbands):
        #variáveis para gerar as estatísticas de ds1 e ds2
        band_ax = ds1.GetRasterBand(b+1)
        band_bx = ds2.GetRasterBand(b+1)
        #constroi o numpy array para fazer a diferença nas bandas
        band_a = ds1.GetRasterBand(b+1).ReadAsArray()
        band_b = ds2.GetRasterBand(b+1).ReadAsArray()
        #transforma para float
        band_a = band_a.astype(float)
        band_b = band_b.astype(float)
        #define -9999 para nan 
        band_a[band_a== -9999]=np.nan
        band_b[band_b== -9999]=np.nan
        #calcula a diferença para a banda
        band_diff = band_a - band_b
        #escreve a diferença no array
        dataset.GetRasterBand(b+1).WriteArray(band_diff)
        #obtem as estatísticas do dataset1 e dataset2
        stats_a = band_ax.GetStatistics(True, True)
        stats_b = band_bx.GetStatistics(True, True)
        #printa as estatísticas das bandas do dataset1 e dataset 2
        print("[ STATS 255] =  Band=%.1d, Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % ((b+1), stats_a[0], stats_a[1], stats_a[2], stats_a[3] ))
        print("[ STATS 280] =  Band=%.1d, Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % ((b+1), stats_b[0], stats_b[1], stats_b[2], stats_b[3] ))


def process_files(type_):
    '''doc
    '''
    filename = "files_{}.json".format(type_)

    files_to_process = {}
    with open(filename) as infile:
        files_to_process = json.load(infile)

    for dirname in files_to_process:
        print('dirname: {}'.format(dirname))
        for files in files_to_process[dirname]:
            print('work: {}'.format(files[0]))
            print('ref: {}'.format(files[1]))
            print('---')
            im_reference = files[1]
            im_target    = files[0]
            kwargs = {
                'grid_res'     : 100,
                'window_size'  : (64,64),
                'path_out'     : files[0],
                'projectDir'   : '/home/marujo/Desktop/CBERS/toa_org',
                'q'            : False,
                'fmt_out'      : "GTIFF",
            }
            CRL = COREG_LOCAL(im_reference,im_target,**kwargs)
            CRL.correct_shifts()
        print('===')


def stack_virtual_raster(image_patter, bands, output):
    band_list = list()
    for band in bands:
        img_path = image_patter.format(band)
        band_list.append(img_path)

    #Set Virtual Raster options
    vrt_options = gdal.BuildVRTOptions(separate='-separate')
    #Create virtual raster
    ds = gdal.BuildVRT(output, band_list, options=vrt_options)
    return ds


def warp(ds):
    ### Geotrans and projections
    geotrans, prj = ds.GetGeoTransform(), ds.GetProjection()

    ### Warp
    gdaloptions = {'format':'VRT', 'srcSRS':prj, 'dstSRS':dst_epsg, 'xRes':geotrans[1], 'yRes':geotrans[5]}
    ds = gdal.Warp('', ds, **gdaloptions)

    return ds


def load_reftar_singband_geoarray(band):
    ### Open Gdal Dataset
    ds_ref = gdal.Open(img_reference.format(band))
    ds_targ = gdal.Open(img_target.format(band))

    ### Warp
    ds_ref = warp(ds_ref)
    ds_targ = warp(ds_targ)

    ### Array, Geotrans and Projections
    ref_array, ref_geotrans, ref_prj = ds_ref.ReadAsArray(), ds_ref.GetGeoTransform(), ds_ref.GetProjection()
    targ_array, targ_geotrans, targ_prj = ds_targ.ReadAsArray(), ds_targ.GetGeoTransform(), ds_targ.GetProjection()

    ### Load into GeoArray
    ref_geoArr = geoarray.GeoArray(ref_array, ref_geotrans, ref_prj)
    targ_geoArr = geoarray.GeoArray(targ_array, targ_geotrans, targ_prj)

    del ds_ref, ds_targ
    return ref_geoArr, targ_geoArr


def load_reftar_multband_geoarray():
    ### Create Virtual Raster
    ds_ref = stack_virtual_raster(img_reference, bands, vrt_ref)
    ds_targ = stack_virtual_raster(img_target, bands, vrt_targ)

    ### Warp
    ds_ref = warp(ds_ref)
    ds_targ = warp(ds_targ)

    ### Array, Geotrans and Projections
    ref_array, ref_geotrans, ref_prj = ds_ref.ReadAsArray(), ds_ref.GetGeoTransform(), ds_ref.GetProjection()
    targ_array, targ_geotrans, targ_prj = ds_targ.ReadAsArray(), ds_targ.GetGeoTransform(), ds_targ.GetProjection()

    ### Load into GeoArray
    ref_geoArr = geoarray.GeoArray(numpy.transpose(ref_array, (1,2,0)), ref_geotrans, ref_prj) #transpose due to geoarray using wrong gdal dimensions
    targ_geoArr = geoarray.GeoArray(numpy.transpose(targ_array, (1,2,0)), targ_geotrans, targ_prj) #transpose due to geoarray using wrong gdal dimensions

    del ds_ref, ds_targ
    return ref_geoArr, targ_geoArr
