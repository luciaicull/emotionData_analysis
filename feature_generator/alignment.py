'''
Make sure the xml-midi or midi-midi pair to be aligned
by using Nakamura Alignment Tool

After passing through AlignmentTool,
match result files(.txt files) will be produced.
 ex) score(xml) - midi : _match.txt 
 ex) midi - midi : _corresp.txt
'''

from pathlib import Path
import shutil
import os
import subprocess

from .constant import ALIGN_DIR
'''
# for debugging
from constant import ALIGN_DIR
'''
class AlignmentTool:
    def __init__(self, ref_path, target_path_list):
        '''
        Parameters 
            ref_path : xml path(in xml-midi) or E1 midi path(in midi-midi). type=Path()
            targets : E1 ~ E5 midi paths(in xml-midi) or E2 ~ E5 midi path(in midi-midi). type=list() of Path()
        '''
        self.ref_path = ref_path
        self.target_path_list = target_path_list

class XmlMidiAlignmentTool(AlignmentTool):
    def __init__(self, ref_path, target_path_list):
        super().__init__(ref_path, target_path_list)
    
    def align(self):
        match_file_path_list = []

        #print("Processing reference: {}".format(self.ref_path.name[:-len('.E1.mid')]))
        for target_path in self.target_path_list:
            #print("Processing target: {}".format(target_path.name[:-len('.E1.mid')]))
            
            aligned_file_name = target_path.name[:-len('.mid')] + '_match.txt'
            aligned_file_path = target_path.parent.joinpath(aligned_file_name)
            
            if not aligned_file_path.exists():
                self._align_with_nakamura(self.ref_path, target_path, aligned_file_path)
            
            match_file_path_list.append(aligned_file_path)
        
        return match_file_path_list

    def _align_with_nakamura(self, ref_path, target_path, aligned_file_path):
        align_tool_path = Path(ALIGN_DIR)
        shutil.copy(ref_path, align_tool_path.joinpath("ref.xml"))
        shutil.copy(target_path, align_tool_path.joinpath("target.mid"))

        current_dir = os.getcwd()

        os.chdir(ALIGN_DIR)
        subprocess.check_call(
            ["sudo", "sh", "MusicXMLToMIDIAlign.sh", "ref", "target"])
        
        shutil.move('target_match.txt', aligned_file_path)

        os.chdir(current_dir)

        

class MidiMidiAlignmentTool(AlignmentTool):
    def __init__(self, ref_path, target_path_list):
        super().__init__(ref_path, target_path_list)
    
    def align(self):
        corresp_file_path_list = []

        #print("Processing reference: {}".format(self.ref_path.name[:-len('.E1.mid')]))
        for target_path in self.target_path_list:
            #print("Processing target: {}".format(target_path.name[:-len('.E1.mid')]))
            
            aligned_file_name = target_path.name[:-len('.mid')] + '_corresp.txt'
            aligned_file_path = target_path.parent.joinpath(aligned_file_name)
            
            if not aligned_file_path.exists():
                self._align_with_nakamura(self.ref_path, target_path, aligned_file_path)
            
            corresp_file_path_list.append(aligned_file_path)
        
        return corresp_file_path_list

    def _align_with_nakamura(self, ref_path, target_path, aligned_file_path):
        align_tool_path = Path(ALIGN_DIR)
        shutil.copy(ref_path, align_tool_path.joinpath("ref.mid"))
        shutil.copy(target_path, align_tool_path.joinpath("target.mid"))

        current_dir = os.getcwd()

        #try:
        os.chdir(ALIGN_DIR)
        subprocess.check_call(
            ["sudo", "sh", "MIDIToMIDIAlign.sh", "ref", "target"])
        #except:
        #    print('Error to process {}'.format(target_path))

        #aligned_file_name = target_path.name[:-len('.mid')] + '_corresp.txt'
        #aligned_file_path = target_path.parent.joinpath(aligned_file_name)

        shutil.move('target_corresp.txt', aligned_file_path)

        os.chdir(current_dir)
        



if __name__ == "__main__":
    ref_path = Path("/home/yoojin/data/emotionDataset/final/tmp_test/Tchaikovsky.the-seasons_op37a_no4_.mm_25-58.s008.E1.mid")
    target_paths = [Path("/home/yoojin/data/emotionDataset/final/tmp_test/Tchaikovsky.the-seasons_op37a_no4_.mm_25-58.s008.E1.mid")]
    
    tool = MidiMidiAlignmentTool(ref_path, target_paths)
    tool.align()
