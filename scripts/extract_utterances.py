import os
from glob import glob
from tqdm import tqdm
import csv

from utils import get_utterances_textgrid, get_partial_audio, write_wave, preprocess_text


def __parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Prepare audio files and transcripts for dataset creation')
    parser.add_argument('--audio_path', help='Folder containing audio files')
    parser.add_argument('--transcript_path', help='Folder containing TextGrid transcripts')
    parser.add_argument('--output_path', help='Output folder')
    return parser.parse_args()


def main():
    args = __parse_args()

    # Create the dataset directory structure
    dataset_path = os.path.join(args.output_path, 'primock57')
    data_folder = os.path.join(dataset_path, 'data')
    os.makedirs(data_folder, exist_ok=True)
    
    readme_path = os.path.join(dataset_path, 'README.md')
    metadata_path = os.path.join(dataset_path, 'metadata.csv')

    # Write a simple README.md
    with open(readme_path, 'w') as f:
        f.write("# My Dataset\n\nThis dataset contains audio files and corresponding transcriptions.\n")

    # Find all recordings in the audio folder and match them with transcripts
    audio_files = glob(f'{args.audio_path}/*.wav')
    transcript_files = glob(f'{args.transcript_path}/*.TextGrid')

    all_utterances = []
    for af in tqdm(audio_files, desc='Processing audio files'):
        c_id = os.path.splitext(os.path.basename(af))[0]
        single_transcript_path = f'{args.transcript_path}/{c_id}.TextGrid'
    
        assert single_transcript_path in transcript_files, f"Missing transcript for {af}"
        utterances = get_utterances_textgrid(single_transcript_path)

        for idx, u in enumerate(utterances):
            utt_id = f"{c_id}_u{idx}"
            utt_audio_path = os.path.join(data_folder, f'{utt_id}.wav')

            cleaned_text = preprocess_text(u['text'])
            # import pdb; pdb.set_trace()
            if len(cleaned_text) > 0:
                utt_audio = get_partial_audio(af, u['from'], u['to'])
                write_wave(utt_audio_path, utt_audio)
                all_utterances.append({
                    'file_name': os.path.join("data", f'{utt_id}.wav'),
                    'transcription': cleaned_text,
                    'utterance': utt_id
                })

    # Write the metadata.csv file
    with open(metadata_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_name', 'transcription', 'utterance']  # Add additional fields as needed
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for utt in all_utterances:
            writer.writerow(utt)

    print(f"Dataset prepared successfully in {dataset_path}")


if __name__ == '__main__':
    main()


