# PNSN's version of Akash's event classifier.

Akash's main repo that this is based on [https://github.com/Akashkharita/Surface_Event_Detection](https://github.com/Akashkharita/Surface_Event_Detection)

# Differences between Akash's and Alex's repos
There are a few differences including:
* how SNR is calculated- ratio of 98th percentile abs(amplitude) in noise and signal window.
* run_all_models.py is a script that runs 4 DL models and 1 ML model (40sec) given an evid. It fetches data from the 10 stations with the earliest pick times regardless of if they are P or S picks, i.e. the closest stations with picks.
* special treatment is given for SU events that have only one station picked and no location.  The event location is assumed to be the station location.  An empirical volcano-specific station list based on stations with the most historical SU picks is used to form the list of 10 stations.
* output are files look like:

62063646_output.txt:

```
ORDATE START END: 62063646 2024-12-02 17:43:27.330000 2024-12-02T17:42:57.330000Z 2024-12-02T17:45:18.330000Z
PROBS: 62063646 SeismicCNN_1d    0 0.9026700 0.3833518 0.7905207 0.9897208  109.18  [['CC.CPCO..BHE', 'CC.CPCO..BHN', 'CC.CPCO..BHZ']] 
PROBS: 62063646 SeismicCNN_1d    1 0.3968616 0.4153720 0.6015000 0.9851815   22.93  [['CC.NORM..BHE', 'CC.NORM..BHN', 'CC.NORM..BHZ']] 
PROBS: 62063646 SeismicCNN_1d    2 0.3564669 0.7930581 0.4278859 0.9963425   13.41  [['CC.CIHL..BHE', 'CC.CIHL..BHN', 'CC.CIHL..BHZ']] 
62063646   SeismicCNN_1d_mean_all              EQ: 0.552  3    EX: 0.531  3    SU: 0.990  3      ProbDist: 0.438    Pred: SU  0.990  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_1d_p0.30_d0.00_snr00     EQ: 0.552  3    EX: 0.531  3    SU: 0.990  3      ProbDist: 0.438    Pred: SU  0.990  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_1d_p0.30_d0.00_snr01     EQ: 0.552  3    EX: 0.531  3    SU: 0.990  3      ProbDist: 0.438    Pred: SU  0.990  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_1d_p0.30_d0.00_snr02     EQ: 0.552  3    EX: 0.531  3    SU: 0.990  3      ProbDist: 0.438    Pred: SU  0.990  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_1d_p0.30_d0.00_snr03     EQ: 0.552  3    EX: 0.531  3    SU: 0.990  3      ProbDist: 0.438    Pred: SU  0.990  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_1d_p0.30_d0.00_snr04     EQ: 0.552  3    EX: 0.531  3    SU: 0.990  3      ProbDist: 0.438    Pred: SU  0.990  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_1d_p0.30_d0.00_snr10     EQ: 0.552  3    EX: 0.531  3    SU: 0.990  3      ProbDist: 0.438    Pred: SU  0.990  3    Analyst: su  Mag: Md2.2
etc
PROBS: 62063646 SeismicCNN_2d    0 0.0931419 0.1001849 0.1111061 0.9978759  109.18  [['CC.CPCO..BHE', 'CC.CPCO..BHN', 'CC.CPCO..BHZ']] 
PROBS: 62063646 SeismicCNN_2d    1 0.0010403 0.0135124 0.1175617 0.9998946   22.93  [['CC.NORM..BHE', 'CC.NORM..BHN', 'CC.NORM..BHZ']] 
PROBS: 62063646 SeismicCNN_2d    2 0.0091292 0.1496502 0.2605225 0.9944595   13.41  [['CC.CIHL..BHE', 'CC.CIHL..BHN', 'CC.CIHL..BHZ']] 
62063646   SeismicCNN_2d_mean_all              EQ: 0.034  3    EX: 0.088  3    SU: 0.997  3      ProbDist: 0.910    Pred: SU  0.997  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_2d_p0.30_d0.00_snr00     EQ: 0.000  0    EX: 0.000  0    SU: 0.997  3      ProbDist: 0.997    Pred: SU  0.997  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_2d_p0.30_d0.00_snr01     EQ: 0.000  0    EX: 0.000  0    SU: 0.997  3      ProbDist: 0.997    Pred: SU  0.997  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_2d_p0.30_d0.00_snr02     EQ: 0.000  0    EX: 0.000  0    SU: 0.997  3      ProbDist: 0.997    Pred: SU  0.997  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_2d_p0.30_d0.00_snr03     EQ: 0.000  0    EX: 0.000  0    SU: 0.997  3      ProbDist: 0.997    Pred: SU  0.997  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_2d_p0.30_d0.00_snr04     EQ: 0.000  0    EX: 0.000  0    SU: 0.997  3      ProbDist: 0.997    Pred: SU  0.997  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_2d_p0.30_d0.00_snr10     EQ: 0.000  0    EX: 0.000  0    SU: 0.997  3      ProbDist: 0.997    Pred: SU  0.997  3    Analyst: su  Mag: Md2.2
62063646   SeismicCNN_2d_p0.30_d0.02_snr00     EQ: 0.000  0    EX: 0.000  0    SU: 0.997  3      ProbDist: 0.997    Pred: SU  0.997  3    Analyst: su  Mag: Md2.2
etc
```

**Columns:**
```ORDATE START END: 62063646 2024-12-02 17:43:27.330000 2024-12-02T17:42:57.330000Z 2024-12-02T17:45:18.330000Z```
evid, origin time, start_time, end_time.

```PROBS: 62063646 SeismicCNN_1d    0 0.9026700 0.3833518 0.7905207 0.9897208  109.18  [['CC.CPCO..BHE', 'CC.CPCO..BHN', 'CC.CPCO..BHZ']]```

evid, model name, station index, EQ probability, EX probability, NOise probability, SU probability, SNR, NSLCs used.

```62063646   SeismicCNN_2d_p0.30_d0.00_snr04     EQ: 0.000  0    EX: 0.000  0    SU: 0.997  3      ProbDist: 0.997    Pred: SU  0.997  3    Analyst: su  Mag: Md2.2```

evid, model_parameter_set, max probability for each of the 3 event classes, probability distance, the event-level prediction and it's probability, N traces used, Analyst label, Catalog magnitude.

*model_parameter_set*: this dictates which traces get to vote on the event-level classification for that set = min probablity threshold, min probability distance threshold, SNR.  SNR is ratio of preP to postP using the 98th percentile abs(amplitude).  Probaility thresholds: [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99], probability distance thresholds: [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5], SNR thresholds: [0, 1, 2, 3, 4, 10].  So 672 combinations were tested plus the simple mean of all stations which for the best performing models, was either the best performer or very close.


Probability Distance: the probability of the event class with highest probability minus that of the second place event class.

# Notes
* **Pick time used for each trace:** This analysis does not use the actual pick time, rather it uses the estimated arrival time based on distance from origin to station and assumes a Vp of 6km/sec.  This is sufficent since the models were trained with some wiggling around the pick time applied (from -20 to +5 seconds).

* **Time windows:** the deep learning models (QuakeXNet 1d/2d and SeismicCNN 1d/2d) use a 100 second long time window for analysis.  The classic ML model, ML40s, uses a 40 second long window.  Each trace is assessed at multiple time shifts around the P wave from 30s before to 10s after with strides/shifts of 5s.  The assigned probability for a given event class is the maximum across any of those 9 windows.

* **Stations selected:** for all EQ and EX events, and SU events with a proper source location (n=187), the 10 stations with the earliest picks were selected, regardless of if it was a P or S pick.  For the vast majority of SU events, there is only one station picked and the location is assumed to be at that location.  For these events, a separate function (get_volcano_stations.py) is used to form an empirical list of up to 10 NET.STAs based on those stations nearest the summit, when the station came online, and which have the most historical picks is used.

* **Channel selection:** for stations with multiple channels, the order of preference is HH, BH, EH, HN, EN.  For 6 channel HH + EN stations, HHZ + HHN + HHN get used.  If a station is a 4 channel station (EHZ, ENZ, ENN, ENE), then the channels that get selected are EHZ + ENN + ENE.  For single channel short periods, the channels are EHZ + EHZ + EHZ since the classifier requries three components.

* **Channel selection bias:** There is no explicit bias for channel types in this data set, however SU events are on volcanoes which are almost all HH and BH 3C stations with a few EHZ tossed in, while EX events often are in rural areas which results in larger distances and a disporportionate amount of strong motion stations.

* **run_all_models.py runtime**: on a modest 2019 linux box takes about 10-15 seconds to load models and download data.  Each of the DL models takes about 1 sec and the ML model takes of order 10 sec (when run across 9 time shifts).

