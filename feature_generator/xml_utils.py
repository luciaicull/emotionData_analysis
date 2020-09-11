import copy

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
    absolute_dynamics, relative_dynamics, cresciutos = dir_enc.get_dynamics(
        directions)
    absolute_dynamics_position = [
        dyn.xml_position for dyn in absolute_dynamics]
    absolute_tempos, relative_tempos = dir_enc.get_tempos(directions)
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
