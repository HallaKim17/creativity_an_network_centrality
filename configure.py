dataset_dir = '../../dataset_git/Dataset'
dataInfo_dir = './data/info'
dataInst_dir = './data/preprocessed_data'
result_dir = './data/result'
figure_melody_dir = './data/figure/melody'
figure_rhythm_dir = './data/figure/rhythm'

dataInfo_filename = 'dataset_info.xlsx'
dataInst_filename = 'dataInst.pkl'
dataInst_rhythm_filename = 'dataInst_rhythm.pkl'
similarity_filename = 'weighted_edge.pkl'
similarity_rhythm_filename = 'weighted_edge_rhythm.pkl'
creativity_score_filename = 'creativity_score.pkl'
creativity_score_rhythm_filename = 'creativity_score_rhythm.pkl'
analysis_filename = 'data_analysis_1000_random_networks.xlsx'
analysis_filename_rhythm = 'data_analysis_rhythm.xlsx'

L_min = 10
L_max = 140
L_step = 10
max_iter = 100000
converge_error = 0.01
balancing_percentile = 50
number_of_random_networks = 1000

LOW = 400
HIGH = 400

song_length_bin = 6

Baroque_start = 1601
Baroque_end = 1750
Classical_start = 1750
Classical_end = 1820
Romantic_start = 1800
Romantic_end = 1859
Post_romantic_start = 1860
Post_romantic_end = 1910
Modern_start = 1890
Modern_end = 1975
Twentieth_century_start = 1901
Twentieth_century_end = 2000

composers = ['Couperin','Telemann','Handel','Bach','D.Scarlatti','Haydn','Clementi','Mozart','Beethoven',
             'Schubert','Chopin','Schumann','Liszt','Brahms','Saint_Saens','Mussorgsky','Tchaikovsky',
             'Dvorak','Grieg','Rimsky_Korsakov','Elgar','Albeniz','Scriabin','Rachmaninoff',
             'Faure','Debussy','Sibelius','Busoni','Satie','Scriabin','Schoenberg','Ravel','Bartok',
             'Stravinsky','Webern','Berg','Prokofiev','Gershwin','Copland','Shostakovich','Messiaen']
eras = ['Baroque', 'Classical', 'Romantic', 'Post-romantic', 'Modern', '20th-century']
Baroque_composers = ['Couperin','Telemann','Handel','Bach','D.Scarlatti']
Classical_composers = ['Haydn','Clementi','Mozart','Beethoven']
Romantic_composers = ['Beethoven','Schubert','Chopin','Schumann','Liszt','Brahms','Saint_Saens','Mussorgsky','Tchaikovsky',
                      'Dvorak','Grieg','Rimsky_Korsakov','Elgar','Albeniz']
Post_romantic_composers = ['Saint_Saens','Dvorak','Grieg','Rimsky_Korsakov','Elgar','Albeniz','Scriabin','Rachmaninoff']
Modern_composers = ['Faure','Debussy','Sibelius','Busoni','Satie','Scriabin','Schoenberg','Ravel','Bartok',
             'Stravinsky','Webern','Berg','Prokofiev','Gershwin','Copland','Shostakovich','Messiaen']
Twentieth_composers = ['Saint_Saens','Elgar','Debussy','Sibelius','Busoni','Satie','Scriabin','Rachmaninoff','Schoenberg','Ravel','Bartok',
             'Stravinsky','Berg','Prokofiev','Copland','Shostakovich','Messiaen']

