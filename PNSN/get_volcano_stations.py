#!/home/ahutko/miniconda3/envs/surface_dl/bin/python

def get_volcano_stations(lat, lon, netstas, startdate):
    """
    Helper function to get empirical netstas list for SU events at PNSN volcanos
    based on those stations nearest the summit and which have the most 
    historical picks.
    
    Args:
        lat (float): Event latitude
        lon (float): Event longitude
        netstas (list): Current list of NET.STA (e.g. ['CC.ARAT', 'CC.ASBU'])
        startdate (datetime): Event start date
    
    Returns:
        list: Updated netstas list with relevant volcano stations
    """
    # Import math module only when needed
    import math

    # Modulate the travel time by fudging the distnace to a longer distance
    #  P waves are assumed to travel 6km/s.  Surface events at something
    #  like 2km/s, So fudge_factor_Vp_Vsurface = 6/2 = 3 in this case.
    fudge_factor_Vp_Vsurface = 3

    dists_km = []
    volcano_location = {}
    # Read in long file of all stations w start/end dates
    station_info = {}
    f = open('channels_squacids_west_coast')
    lines = f.readlines()
    f.close()
    
    # Process station channel info
    for line in lines:
        parts = line.split()
        if len(parts) < 8:  # Make sure line has enough fields
            continue
        
        sncl = parts[0]
        sncl_parts = sncl.split('.')
        if len(sncl_parts) != 4:
            continue
            
        net = sncl_parts[0]
        sta = sncl_parts[1]
        loc = sncl_parts[2]  # Location code (-- or 01, etc.)
        channel = sncl_parts[3]
        
        channel_start = parts[2]
        channel_end = parts[3]
        sta_lat = float(parts[4])
        sta_lon = float(parts[5])
        
        # Store info by station
        if sta not in station_info:
            station_info[sta] = {
                'channels': {},
                'lat': sta_lat,
                'lon': sta_lon,
                'net': net,
                'earliest_start': channel_start,
                'latest_end': channel_end
            }
        else:
            # Update earliest start date if this channel has an earlier date
            if channel_start < station_info[sta]['earliest_start']:
                station_info[sta]['earliest_start'] = channel_start
                
            # Update latest end date if this channel has a later date
            if channel_end > station_info[sta]['latest_end']:
                station_info[sta]['latest_end'] = channel_end
        
        # Store channel info with start/end dates
        channel_key = f"{loc}.{channel}"
        station_info[sta]['channels'][channel_key] = {
            'start': channel_start, 
            'end': channel_end
        }
    
    # Empirical list of PNSN volcano stations
    mylist = 'ARAT,ASBU,BRSP,CARB,CIHL,COPP,CPCO,CRBN,GNOB,HIYU,HOA,HUSB,JRO,KWBU,KWBU,LOO,LSON,MILD,NORM,OBSR,OPCH,PALM,PALM,PANH,PARA,PR05,PRLK,REM,RUSH,SEP,SEP,SHRK,SIFT,STD,STD,SUG,SUG,SVIC,SVIC,SWF2,SWNB,TABR,TAVI,TCBU,TIMB,TIMB,TMBU,UNFR,USFR,VALT,VOIT,WIFE,WOW,YOCR,VDEB,PINE,PINE,BERY,BHAM,CDF,DONK,EDM,EDM,ETW,FL2,FMW,FMW,GPW,GPW,HANS,HDW,HOOD,HSR,HSR,JCW,JUN,LO2,LON,LON,MBW,MBW2,MOON,MULN,NCO,NCO,NN19,NN21,OLGA,PASS,RCM,RCS,RER,RER,RPW2,RVC,SAXON,SHUK,SHW,SHW,SLF,SOS,STAR,STAR,TDH,TDH,TDL,TURTL,TWISP,VLL,WRW'
    stations = mylist.split(',')
    
    # PNSN volcano locations
    volcano_locations = {}
    volcano_location['sister'] = [44.104, -121.770]
    volcano_location['newberry'] = [43.725, -121.234]
    volcano_location['hood'] = [45.37, -121.69]
    volcano_location['helens'] = [46.20, -122.19]
    volcano_location['baker'] = [48.77, -121.82]
    volcano_location['rainier'] = [46.85, -121.76]
    
    # PNSN list of stations w most historical SU picks at volcanos
    volcano_stations = {}
    volcano_stations['sisters'] = ['MOON', 'WIFE', 'HUSB', 'PRLK', 'TCBU']
    volcano_stations['newberry'] = ['CIHL', 'CPCO', 'KWBU', 'NORM', 'SVIC', 'SWNB', 'TMBU', 'NCO', 'NN19', 'NN21', 'ASBU']
    volcano_stations['hood'] = ['BRSP', 'LSON', 'PALM', 'TIMB', 'YOCR', 'HOOD', 'TDH']
    volcano_stations['helens'] = ['LOO', 'REM', 'SEP', 'STD', 'SUG', 'SWF2', 'USFR', 'EDM', 'HSR', 'SHW']
    volcano_stations['baker'] = ['SHUK', 'MBW2', 'VDEB', 'BHAM', 'DONK', 'JCW', 'MULN', 'PASS', 'RPW2', 'SAXON']
    volcano_stations['rainier'] = ['RCM', 'RCS', 'RER', 'STAR', 'OBSR', 'PANH', 'SIFT', 'MILD', 'PARA', 'SIFT', 'PR05', 'COPP', 'FMW', 'RUSH', 'LON', 'ARAT']
    
    # Determine nearest volcano
    nearest_volcano = None
    min_distance = float('inf')
    
    for volcano, coords in volcano_location.items():
        v_lat, v_lon = coords
        # Modulate longitude by the volcano's latitude to account for spherical distance
        lon_factor = math.cos(math.radians(v_lat))
        distance = ((lat - v_lat) ** 2 + ((lon - v_lon) * lon_factor) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            nearest_volcano = volcano
    
    # If we identified a nearby volcano, add its stations
    added_stations = set()  # To avoid duplicates in case of repeats in the list

    if nearest_volcano and nearest_volcano in volcano_stations:
        # Format startdate properly for string comparison
        startdate_str = startdate.strftime("%Y-%m-%dT%H:%M:%S")
        
        # Go through the list of volcano_stations in order and add those that meet criteria
        for station in volcano_stations[nearest_volcano]:
            # Limit to 10 stations total
            if len(netstas) >= 10:
                break
                
            # Skip if we've already added this station
            if station in added_stations:
                continue
                
            if station in station_info:
                added_stations.add(station)
                net = station_info[station]['net']
                
                # Check if station was operational during startdate (using full date range)
                if station_info[station]['earliest_start'] <= startdate_str <= station_info[station]['latest_end']:
                    netsta = f"{net}.{station}"
                    if netsta not in netstas:  # Avoid duplicates
                        netstas.append(netsta)
                    
                    # No need to check channels since we're using the station's full date range

    for netsta in netstas:
        sta = netsta.split('.')[1]
        sta_lat = station_info[sta]['lat']
        sta_lon = station_info[sta]['lon']
        lon_factor = math.cos(math.radians(v_lat))
        distance = ((lat - sta_lat) ** 2 + ((lon - sta_lon) * lon_factor) ** 2) ** 0.5
        distance_km = distance * 111.19
        distance_km = distance_km * fudge_factor_Vp_Vsurface
        dists_km.append(distance_km)

    return netstas, dists_km

