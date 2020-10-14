from pathlib import Path

TEST_LIST = ['Bach.french-suite_bwv812_no1_allemande.mm_1-end.s004',
             'Bach.french-suite_bwv816_no5_courante.mm_1-end.s002',
             'Chopin.nocturne_op9_no2_.mm_1-12.s021',
             'Schumann.fantasiestucke_op12_no3_.mm_1-42.s021',
             'Bach.minuet_bwv114__.mm_1-32.s018',
             'Schubert.impromptu_op142_no3_var1.mm_1-end.s018',
             'Schubert.impromptu_op142_no3_thema.mm_1-end.s018',
             'Schumann.symphonicetudes_op.13_thema.mm_1-end.s018',
             'Chopin.waltz_op69_no1_.mm_1-16.s020',
             'Bach.minuet_bwv114__.mm_1-32.s016',
             'Schumann.kinderszenen_op15_no7_traumerei.mm_1-end.s014',
             'Schumann.davidsbundlertanze_op6_no2_.mm_1-end.s015',
             'Bartok.roumanian-folk-dances_sz56_no3_.mm_1-end.s005',
             'Bartok.roumanian-folk-dances_sz56_no4_.mm_1-end.s005',
             'Bach.prelude-and-fugue_bwv875_no6_prelude.mm_1-end.s022',
             'Rachmaninoff.piano-concerto_op18_no2_mov1.mm_83-103.s022',
             'Chopin.prelude_op28_no4_.mm_1-12.s007',
             'Beethoven.sonata_op31-2_no17_mov2.mm_1-17.s007',
             'Chopin.valse-brillante_op34_no2_.mm_1-36.s007',
             'Chopin.nocturne_op9_no2_.mm_1-12.s007',
             'Liszt.consolation_s172_no3_.mm_1-45.s007',
             'Mozart.sonata_k545_no16_mov3.mm_1-20.s007',
             'Mozart.sonata_k332_no12_mov2.mm_1-8.s007',
             'Beethoven.sonata_op57_no23_mov3.mm_1-67.s007',
             'Chopin.etude_op25_no7_.mm_1-20.s007',
             'Schumann.kinderszenen_op15_no7_traumerei.mm_1-end.s007',
             'Bach.prelude-and-fugue_bwv880_no11_prelude.mm_1-end.s006',
             'Rachmaninoff.sonata_op36_no2_mov1.mm_1-37.s011',
             'Clementi.sonatine_op36_no1_mov2.mm_1-end.s012',
             'Chopin.valse-brillante_op34_no2_.mm_1-36.s008',
             'Chopin.etude_op10_no4_.mm_1-33.s013']

VALID_LIST = ['Bach.french-suite_bwv816_no5_gavotte.mm_1-end.s002',
              'Bach.french-suite_bwv816_no5_gigue.mm_1-end.s002',
              'Bach.partita_bwv825_no1_minuet1-2.mm_1-end.s018',
              'Bach.prelude-and-fugue_bwv883_no14_prelude.mm_1-end.s018',
              'Prokofiev.sonata_op14_no2_mov1.mm_1-127.s018',
              'Schubert.impromptu_op142_no3_var3.mm_1-16.s018',
              'Bach.prelude-and-fugue_bwv870_no1_prelude.mm_1-19.s007',
              'Beethoven.sonata_op31-2_no17_mov3.mm_1-43.s007',
              'Mozart.sonata_k545_no16_mov2.mm_1-16.s007',
              'Bartok.roumanian-folk-dances_sz56_no2_.mm_1-end.s005',
              'Bartok.roumanian-folk-dances_sz56_no5_.mm_1-end.s005',
              'Beethoven.sonata_op10-3_no7_mov1.mm_1-125.s011',
              'Mozart.nine-variations-on-a-lison-dormait_k573__.mm_1-72.s011'
              'Rachmaninoff.sonata_op36_no2_mov2.mm_1-23.s011',
              'Chopin.barcarolle_op60__.mm_1-38.s019',
              'Beethoven.sonata_op27-2_no14_mov1.mm_1-23.s008',
              'Chopin.berceuse_op57__.mm_1-34.s017',
              'Chopin.nocturne_op48_no1_.mm_1-24.s004',
              'Schumann.davidsbundlertanze_op6_no14_.mm_1-end.s015',
              'Schumann.davidsbundlertanze_op6_no18_.mm_1-end.s015',
              'Schumann.davidsbundlertanze_op6_no5_.mm_1-end.s015',
              'Liszt.dante-sonata_s161_no7_.mm_35-54.s022',
              'Mompou.cancion-y-danza__no6_cancion.mm_1-end.s022',
              'Chopin.waltz_op69_no2_.mm_1-16.s012',
              'Beethoven.sonata_op109_no30_mov1.mm_1-59.s019',
              'Schubert.sonata_d850_no17_mov1.mm_1-47.s019',
              'Chopin.etude_op25_no9_.mm_1-37.s013',
              'Mozart.sonata_k332_no12_mov1.mm_1-40.s007']

TEST_FEATURE_KEYS = ['relative_velocity_std', 'relative_velocity_mean', 'velocity_ratio_std',
                     'relative_beat_tempo_mean', 'relative_measure_tempo_mean', 'beat_tempo_ratio_std', 'relative_beat_tempo_std',
                     'original_duration_ratio_std', 'relative_elongated_duration_std', 'relative_original_duration_std',
                     'relative_onset_deviation_std']

FEATURE_KEYS = ['relative_beat_tempo_mean',
                'beat_tempo_ratio_mean',
                'beat_tempo_ratio_std',

                'relative_velocity_mean',
                'relative_velocity_std',
                'relative_velocity_kurt',
                'velocity_ratio_mean',
                'velocity_ratio_std',

                'relative_original_duration_mean',
                'original_duration_ratio_mean'
                #'original_duration_ratio_std',

                #'relative_elongated_duration_mean'
                ]
'''
FEATURE_KEYS = ['beat_tempo_ratio_mean',
                'beat_tempo_ratio_std',
                'relative_beat_tempo_mean',

                'velocity_ratio_mean',
                'velocity_ratio_std',
                'relative_velocity_mean',
                'relative_velocity_std',
                'relative_velocity_kurt',

                'original_duration_ratio_mean',
                'relative_original_duration_mean']
'''

STAT_TYPE = 'total_scaled_statistics'

BATCH_SIZE = 1
NUM_EPOCH = 100
LEARNING_RATE = 1e-3

DATA_PATH = Path('/home/yoojin/data/emotionDataset/data_for_analysis/entire_dataset')
FILE_NAME = 'feature_dict_for_analysis_with_stats_0809_each_song.dat'
