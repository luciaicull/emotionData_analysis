from . import utils


class Tempo:
    def __init__(self, xml_position, qpm, time_position, end_xml, end_time):
        self.qpm = qpm
        self.xml_position = xml_position
        self.time_position = time_position
        self.end_time = end_time
        self.end_xml = end_xml

    def __str__(self):
        string = '{From ' + str(self.xml_position)
        string += ' to ' + str(self.end_xml)
        return string

def _cal_tempo_by_positions(positions, position_pairs):
        """ Returns list of Tempo objects

        Args:
            positions (1-D list): list of xml positions in piece (ex. beat, measure)
            position_pairs (1-D list): list of valid pair dictionaries {'xml_position', 'time_position', 'pitch', 'index', 'divisions'}
        
        Returns:
            tempos (1-D list): list of .Tempo object
        
        Example:
            (in feature_extraction.py -> PerformExtractor().get_beat_tempo())
            >>> tempos = feature_utils.cal_tempo_by_positions(piece_data.beat_positions, perform_data.valid_position_pairs)
            
        """
        tempos = []
        num_positions = len(positions)
        previous_end = 0

        for i in range(num_positions-1):
            position = positions[i]
            current_pos_pair = get_item_by_xml_position(position_pairs, position)
            if current_pos_pair['xml_position'] < previous_end:
                continue

            next_position = positions[i+1]
            next_pos_pair = get_item_by_xml_position(position_pairs, next_position)

            if next_pos_pair['xml_position'] == previous_end:
                continue

            if current_pos_pair == next_pos_pair:
                continue

            cur_xml = current_pos_pair['xml_position']
            cur_time = current_pos_pair['time_position']
            cur_divisions = current_pos_pair['divisions']
            next_xml = next_pos_pair['xml_position']
            next_time = next_pos_pair['time_position']
            qpm = (next_xml - cur_xml) / \
                (next_time - cur_time) / cur_divisions * 60

            if qpm > 1000:
                print('need check: qpm is ' + str(qpm) + ', current xml_position is ' + str(cur_xml))
            tempo = Tempo(cur_xml, qpm, cur_time, next_xml, next_time)
            tempos.append(tempo)        #
            previous_end = next_pos_pair['xml_position']

        return tempos


def get_item_by_xml_position(alist, item):
    if hasattr(item, 'xml_position'):
        item_pos = item.xml_position
    elif hasattr(item, 'note_duration'):
        item_pos = item.note_duration.xml_position
    elif hasattr(item, 'start_xml_position'):
        item_pos = item.start.xml_position
    elif isinstance(item, dict) and 'xml_position' in item:
        item_pos = item['xml_position']
    else:
        item_pos = item

    repre = alist[0]

    if hasattr(repre, 'xml_position'):
        pos_list = [x.xml_position for x in alist]
    elif hasattr(repre, 'note_duration'):
        pos_list = [x.note_duration.xml_position for x in alist]
    elif hasattr(repre, 'start_xml_position'):
        pos_list = [x.start_xml_position for x in alist]
    elif isinstance(repre, dict) and 'xml_position' in repre:
        pos_list = [x['xml_position'] for x in alist]
    else:
        pos_list = alist

    index = utils.binary_index(pos_list, item_pos)

    return alist[index]
