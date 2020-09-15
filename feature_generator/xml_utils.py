import copy

absolute_tempos_keywords = ['adagio', 'grave', 'lento', 'largo', 'larghetto', 'andante', 'andantino', 'moderato',
                            'allegretto', 'allegro', 'vivace', 'accarezzevole', 'languido', 'tempo giusto', 'mesto',
                            'presto', 'prestissimo', 'maestoso', 'lullaby', 'doppio movimento', 'agitato', 'precipitato',
                            'leicht und zart', 'aufgeregt', 'bewegt', 'rasch', 'innig', 'lebhaft', 'geschwind',
                            "d'un rythme souple",
                            'lent', 'large', 'vif', 'animé', 'scherzo', 'menuetto', 'minuetto']
relative_tempos_keywords = ['animato', 'pesante', 'veloce', 'agitato',
                            'acc', 'accel', 'rit', 'ritard', 'ritardando', 'accelerando', 'rall', 'rallentando', 'ritenuto', 'string',
                            'a tempo', 'im tempo', 'stretto', 'slentando', 'meno mosso', 'meno vivo', 'più mosso', 'allargando', 'smorzando', 'appassionato', 'perdendo',
                            'langsamer', 'schneller', 'bewegter',
                            'retenu', 'revenez', 'cédez', 'mesuré', 'élargissant', 'accélerez', 'rapide', 'reprenez  le  mouvement']
relative_long_tempo_keywords = ['meno mosso', 'meno vivo', 'più mosso', 'animato', 'langsamer', 'schneller',
                                'stretto', 'bewegter', 'tranquillo', 'agitato', 'appassionato']
tempo_primo_words = ['tempo i', 'tempo primo', 'erstes tempo', '1er mouvement',
                     '1er mouvt', 'au mouvtdu début', 'au mouvement', 'au mouvt', '1o tempo']
absolute_dynamics_keywords = ['pppp', 'ppp', 'pp', 'p',
                              'piano', 'mp', 'mf', 'f', 'forte', 'ff', 'fff', 'fp', 'ffp']
relative_dynamics_keywords = ['crescendo', 'diminuendo', 'cresc', 'dim', 'dimin' 'sotto voce',
                              'mezza voce', 'sf', 'fz', 'sfz', 'rf,' 'sffz', 'rf', 'rinf',
                              'con brio', 'con forza', 'con fuoco', 'smorzando', 'appassionato', 'perdendo']


def apply_directions_to_notes(xml_notes, directions, time_signatures):
    """ apply xml directions to each xml_notes

    Args:
        xml_notes (1-D list): list of Note() object in xml of shape (num_notes, )
        directions (1-D list): list of Direction() object in xml of shape (num_direction, )
        time_signatures (1-D list): list of TimeSignature() object in xml of shape (num_time_signature, )

    Returns:
        xml_notes (1-D list): list of direction-encoded Note() object in xml of shape (num_notes, )

    Example:
        (in data_class.py -> _get_direction_encoded_notes())
        >>> notes, rests = self.xml_obj.get_notes()
        >>> directions = self.xml_obj.get_directions()
        >>> time_signatures = self.xml_obj.get_time_signatures()
        >>> self.xml_notes = xml_utils.apply_directions_to_notes(notes, directions, time_signatures)
        >>> self.num_notes = len(self.xml_notes)
    """
    absolute_dynamics, relative_dynamics, cresciutos = get_dynamics(directions)
    absolute_dynamics_position = [dyn.xml_position for dyn in absolute_dynamics]
    absolute_tempos, relative_tempos = get_tempos(directions)
    absolute_tempos_position = [tmp.xml_position for tmp in absolute_tempos]
    time_signatures_position = [time.xml_position for time in time_signatures]

    num_dynamics = len(absolute_dynamics)
    num_tempos = len(absolute_tempos)

    for note in xml_notes:
        note_position = note.note_duration.xml_position

        if num_dynamics > 0:
            index = binary_index(absolute_dynamics_position, note_position)
            note.dynamic.absolute = absolute_dynamics[index].type['content']
            note.dynamic.absolute_position = absolute_dynamics[index].xml_position

        if num_tempos > 0:
            tempo_index = binary_index(absolute_tempos_position, note_position)
        # note.tempo.absolute = absolute_tempos[tempo_index].type[absolute_tempos[tempo_index].type.keys()[0]]
            note.tempo.absolute = absolute_tempos[tempo_index].type['content']
            note.tempo.recently_changed_position = absolute_tempos[tempo_index].xml_position
        time_index = binary_index(time_signatures_position, note_position)
        note.tempo.time_numerator = time_signatures[time_index].numerator
        note.tempo.time_denominator = time_signatures[time_index].denominator
        note.tempo.time_signature = time_signatures[time_index]

        # have to improve algorithm
        for rel in relative_dynamics:
            if rel.xml_position > note_position:
                continue
            if note_position < rel.end_xml_position:
                note.dynamic.relative.append(rel)
                if rel.xml_position > note.tempo.recently_changed_position:
                    note.tempo.recently_changed_position = rel.xml_position

        for cresc in cresciutos:
            if cresc.xml_position > note_position:
                break
            if note_position < cresc.end_xml_position:
                note_cresciuto = note.dynamic.cresciuto
                if note_cresciuto is None:
                    note.dynamic.cresciuto = copy.copy(cresc)
                else:
                    prev_type = note.dynamic.cresciuto.type
                    if cresc.type == prev_type:
                        note.dynamic.cresciuto.overlapped += 1
                    else:
                        if note_cresciuto.overlapped == 0:
                            note.dynamic.cresciuto = None
                        else:
                            note.dynamic.cresciuto.overlapped -= 1

        if len(note.dynamic.relative) > 1:
            note = divide_cresc_staff(note)

        for rel in relative_tempos:
            if rel.xml_position > note_position:
                continue
            if note_position < rel.end_xml_position:
                note.tempo.relative.append(rel)

    return xml_notes


def get_dynamics(directions):
    temp_abs_key = absolute_dynamics_keywords
    temp_abs_key.append('dynamic')

    absolute_dynamics = extract_directions_by_keywords(directions, temp_abs_key)
    relative_dynamics = extract_directions_by_keywords(directions, relative_dynamics_keywords)
    abs_dynamic_dummy = []
    for abs in absolute_dynamics:
        if abs.type['content'] == 'fp':
            abs.type['content'] = 'f'
            abs2 = copy.copy(abs)
            abs2.xml_position += 0.1
            abs2.type = copy.copy(abs.type)
            abs2.type['content'] = 'p'
            abs_dynamic_dummy.append(abs2)
        elif abs.type['content'] == 'ffp':
            abs.type['content'] = 'ff'
            abs2 = copy.copy(abs)
            abs2.xml_position += 0.1
            abs2.type = copy.copy(abs.type)
            abs2.type['content'] = 'p'
            abs_dynamic_dummy.append(abs2)
        elif abs.type['content'] == 'sfp':
            abs.type['content'] = 'sf'
            abs2 = copy.copy(abs)
            abs2.xml_position += 0.1
            abs2.type = copy.copy(abs.type)
            abs2.type['content'] = 'p'
            abs_dynamic_dummy.append(abs2)

        if abs.type['content'] in ['sf', 'fz', 'sfz', 'sffz', 'rf', 'rfz']:
            relative_dynamics.append(abs)
        else:
            abs_dynamic_dummy.append(abs)

    absolute_dynamics = abs_dynamic_dummy
    absolute_dynamics, temp_relative = check_relative_word_in_absolute_directions(
        absolute_dynamics)
    relative_dynamics += temp_relative
    dummy_rel = []
    for rel in relative_dynamics:
        if rel not in absolute_dynamics:
            dummy_rel.append(rel)
    relative_dynamics = dummy_rel

    relative_dynamics.sort(key=lambda x: x.xml_position)
    relative_dynamics = merge_start_end_of_direction(relative_dynamics)
    absolute_dynamics_position = [dyn.xml_position for dyn in absolute_dynamics]
    cresc_name = ['crescendo', 'diminuendo']
    cresciuto_list = []
    num_relative = len(relative_dynamics)

    for i in range(num_relative):
        rel = relative_dynamics[i]
        if len(absolute_dynamics) > 0:
            index = binary_index(absolute_dynamics_position, rel.xml_position)
            rel.previous_dynamic = absolute_dynamics[index].type['content']
            if index + 1 < len(absolute_dynamics):
                # .type['content']
                rel.next_dynamic = absolute_dynamics[index + 1]

            else:
                rel.next_dynamic = absolute_dynamics[index]
        if rel.type['type'] == 'dynamic' and not rel.type['content'] in ['rf', 'rfz', 'rffz']:  # sf, fz, sfz
            rel.end_xml_position = rel.xml_position + 0.1

        if not hasattr(rel, 'end_xml_position'):
            # if rel.end_xml_position is None:
            for j in range(1, num_relative-i):
                next_rel = relative_dynamics[i+j]
                rel.end_xml_position = next_rel.xml_position
                break

        if len(absolute_dynamics) > 0 and hasattr(rel, 'end_xml_position') and index < len(absolute_dynamics) - 1 and absolute_dynamics[index + 1].xml_position < rel.end_xml_position:
            rel.end_xml_position = absolute_dynamics_position[index + 1]

        if not hasattr(rel, 'end_xml_position'):
            rel.end_xml_position = float("inf")

        if (rel.type['type'] in cresc_name or crescendo_word_regularization(rel.type['content']) in cresc_name)\
                and (hasattr(rel, 'next_dynamic') and rel.end_xml_position < rel.next_dynamic.xml_position):
            if rel.type['type'] in cresc_name:
                cresc_type = rel.type['type']
            else:
                cresc_type = crescendo_word_regularization(rel.type['content'])
            cresciuto = Cresciuto(rel.end_xml_position,
                                  rel.next_dynamic.xml_position, cresc_type)
            cresciuto_list.append(cresciuto)

    return absolute_dynamics, relative_dynamics, cresciuto_list


def merge_start_end_of_direction(directions):
    for i in range(len(directions)):
        dir = directions[i]
        type_name = dir.type['type']
        if type_name in ['crescendo', 'diminuendo', 'pedal'] and dir.type['content'] == "stop":
            for j in range(i):
                prev_dir = directions[i-j-1]
                prev_type_name = prev_dir.type['type']
                if type_name == prev_type_name and prev_dir.type['content'] == "start" and dir.staff == prev_dir.staff:
                    prev_dir.end_xml_position = dir.xml_position
                    break
    dir_dummy = []
    for dir in directions:
        type_name = dir.type['type']
        if type_name in ['crescendo', 'diminuendo', 'pedal'] and dir.type['content'] != "stop":
            # directions.remove(dir)
            dir_dummy.append(dir)
        elif type_name == 'words':
            dir_dummy.append(dir)
    directions = dir_dummy
    return directions
    

class Cresciuto:
    def __init__(self, start, end, type):
        self.xml_position = start
        self.end_xml_position = end
        self.type = type  # crescendo or diminuendo
        self.overlapped = 0

def crescendo_word_regularization(word):
    word = word_regularization(word)
    if 'decresc' in word:
        word = 'diminuendo'
    elif 'cresc' in word:
        word = 'crescendo'
    elif 'dim' in word:
        word = 'diminuendo'
    return word


def get_tempos(directions):
    absolute_tempos = extract_directions_by_keywords(
        directions, absolute_tempos_keywords)
    relative_tempos = extract_directions_by_keywords(
        directions, relative_tempos_keywords)
    relative_long_tempos = extract_directions_by_keywords(
        directions, relative_long_tempo_keywords)

    if (len(absolute_tempos) == 0 or absolute_tempos[0].xml_position != 0) \
            and len(relative_long_tempos) > 0 and relative_long_tempos[0].xml_position == 0:
        absolute_tempos.insert(0, relative_long_tempos[0])

    dummy_relative_tempos = []
    for rel in relative_tempos:
        if rel not in absolute_tempos:
            dummy_relative_tempos.append(rel)
    relative_tempos = dummy_relative_tempos

    dummy_relative_tempos = []
    for rel in relative_long_tempos:
        if rel not in absolute_tempos:
            dummy_relative_tempos.append(rel)
    relative_long_tempos = dummy_relative_tempos

    absolute_tempos, temp_relative = check_relative_word_in_absolute_directions(
        absolute_tempos)
    relative_tempos += temp_relative
    relative_long_tempos += temp_relative
    relative_tempos.sort(key=lambda x: x.xml_position)

    absolute_tempos_position = [tmp.xml_position for tmp in absolute_tempos]
    num_abs_tempos = len(absolute_tempos)
    num_rel_tempos = len(relative_tempos)

    for abs in absolute_tempos:
        for wrd in tempo_primo_words:
            if wrd in abs.type['content'].lower():
                abs.type['content'] = absolute_tempos[0].type['content']

    for i, rel in enumerate(relative_tempos):
        if rel not in relative_long_tempos and i+1 < num_rel_tempos:
            rel.end_xml_position = relative_tempos[i+1].xml_position
        elif rel in relative_long_tempos:
            for j in range(1, num_rel_tempos-i):
                next_rel = relative_tempos[i+j]
                if next_rel in relative_long_tempos:
                    rel.end_xml_position = next_rel.xml_position
                    break
        if len(absolute_tempos) > 0:
            index = binary_index(absolute_tempos_position, rel.xml_position)
            rel.previous_tempo = absolute_tempos[index].type['content']
            if index+1 < num_abs_tempos:
                rel.next_tempo = absolute_tempos[index+1].type['content']
                if not hasattr(rel, 'end_xml_position') or rel.end_xml_position > absolute_tempos_position[index+1]:
                    rel.end_xml_position = absolute_tempos_position[index+1]
        if not hasattr(rel, 'end_xml_position'):
            rel.end_xml_position = float("inf")

    return absolute_tempos, relative_tempos


def check_relative_word_in_absolute_directions(abs_directions):
    relative_keywords = ['più', 'meno', 'plus', 'moins', 'mehr', 'bewegter', 'langsamer']
    absolute_directions = []
    relative_directions = []
    for dir in abs_directions:
        dir_word = word_regularization(dir.type['content'])
        for rel_key in relative_keywords:
            if rel_key in dir_word:
                relative_directions.append(dir)
                break
        else:
            absolute_directions.append(dir)

    return absolute_directions, relative_directions


def extract_directions_by_keywords(directions, keywords):
    sub_directions = []

    for dir in directions:
        included = check_direction_by_keywords(dir, keywords)
        if included:
            sub_directions.append(dir)

    return sub_directions


def check_direction_by_keywords(dir, keywords):
    if dir.type['type'] in keywords:
        return True
    elif dir.type['type'] == 'words':
        dir_word = word_regularization(dir.type['content'])
        if dir_word in keywords:
            return True
        else:
            word_split = dir_word.split(' ')
            for w in word_split:
                if w in keywords:
                    return True

        for key in keywords:  # words like 'sempre più mosso'
            if len(key) > 2 and key in dir_word:
                return True


def word_regularization(word):
    if word:
        word = word.replace(',', ' ').replace('.', ' ').replace(
            '\n', ' ').replace('(', '').replace(')', '').replace('  ', ' ').lower()
    else:
        word = None
    return word


def divide_cresc_staff(note):
    """ update note.dynamic

    Args:
        note: Note() object in xml_notes

    Returns:
        note: dynamic updated Note() object in xml_notes

    Example:
        (in apply_directions_to_notes())
        >>> note = divide_cresc_staff(note)
    """
    #check the note has both crescendo and diminuendo (only wedge type)
    cresc = False
    dim = False
    for rel in note.dynamic.relative:
        if rel.type['type'] == 'crescendo':
            cresc = True
        elif rel.type['type'] == 'diminuendo':
            dim = True

    if cresc and dim:
        delete_list = []
        for i in range(len(note.dynamic.relative)):
            rel = note.dynamic.relative[i]
            if rel.type['type'] in ['crescendo', 'diminuendo']:
                if (rel.placement == 'above' and note.staff == 2) or (rel.placement == 'below' and note.staff == 1):
                    delete_list.append(i)
        for i in sorted(delete_list, reverse=True):
            del note.dynamic.relative[i]

    return note


def binary_index(alist, item):
    first = 0
    last = len(alist)-1
    midpoint = 0

    if(item < alist[first]):
        return 0

    while first < last:
        midpoint = (first + last)//2
        currentElement = alist[midpoint]

        if currentElement < item:
            if alist[midpoint+1] > item:
                return midpoint
            else:
                first = midpoint + 1
            if first == last and alist[last] > item:
                return midpoint
        elif currentElement > item:
            last = midpoint - 1
        else:
            if midpoint + 1 == len(alist):
                return midpoint
            while alist[midpoint+1] == item:
                midpoint += 1
                if midpoint + 1 == len(alist):
                    return midpoint
            return midpoint
    return last
