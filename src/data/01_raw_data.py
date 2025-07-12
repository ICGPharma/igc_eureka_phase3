import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import os
import re
from pydub import AudioSegment
from moviepy import VideoFileClip
import argparse

#TODO: MAYBE DO SOMETHING TO DELETE GREEK AND TAIWANESE RECORDINGS, and PITT-org??
#TODO: DePaul no quedó como en nuestros datos.
#TODO: Holland no quedó como en nuestros datos.
#TODO: Protocol/Baycrest & /Baycrest-PPA no quedó como en nuestros datos.
#TODO: Protocol/Delaware tiene más datos que los nuestros.
#TODO: Algunos de WLS no quedaron iguales - Pero son como max 2 por carpeta
def process_audios(dir):
    
    BASE_PATH = '/buckets/projects/ct_igc_phase3/phase3/talkbank_dementia'
    TRANSCRIPT_DIR = '/buckets/projects/ct_igc_phase3/phase3/talkbank_dementia_transcripts'
    OUTPUT_DIR = '/buckets/projects/ct_igc_phase3/phase3/talkbank_dementia_processed'
    FILE_EXTENSIONS = ["mp3", "mp4", "wav"]

    def extract_timestamps(path):
        with open(path) as file:
            past_line=None
            text = []
            for line in file.readlines():
                if line.startswith('*'):
                    text.append(line)
                    past_line=line
                elif not line.startswith('%') and past_line and past_line.startswith('*'):
                    text[-1] = ' '.join([text[-1].strip(),line.strip()])

            speaker_data = []
            for line in text:
                timestamp_pattern = re.compile(r'(\d+)_(\d+)')
                speaker = line.split(':')[0].strip('*')

                timestamps = timestamp_pattern.search(line)

                if timestamps:
                    start_time_ms = int(timestamps.group(1))
                    end_time_ms = int(timestamps.group(2))

                    start_time_sec = start_time_ms
                    end_time_sec = end_time_ms

                    speaker_data.append({
                        'speaker': speaker,
                        'start_time_msec': start_time_sec,
                        'end_time_msec': end_time_sec
                    })
            return speaker_data

    def split_audio(audio_dir, audio_file, timestamps):
        audio_path = os.path.join(audio_dir,audio_file)
        if os.path.exists(os.path.join(OUTPUT_DIR,audio_path)):
            print('[ALREADY DONE]',audio_path)
            return
        
        print('[PROCESSING]',audio_path)
        try:
            participant_segments = [seg for seg in timestamps if seg['speaker'].startswith('PAR')]
            audio = AudioSegment.from_file(os.path.join(BASE_PATH,audio_path))

            output_audio = AudioSegment.empty()
            for seg in participant_segments:
                start_ms = int(seg['start_time_msec'])
                end_ms = int(seg['end_time_msec'])
                output_audio += audio[start_ms:end_ms]
            
            os.makedirs(os.path.join(OUTPUT_DIR,audio_dir), exist_ok=True)

            # Export the final combined audio
            output_audio.export(os.path.join(OUTPUT_DIR,audio_path))
            print('[DONE]',audio_path)
        except Exception as e:
            print('[ERROR] Couldn\'t process audio',audio_path)
            print(e)

    path = os.path.join(BASE_PATH,dir)
    for file in os.listdir(os.path.join(BASE_PATH,dir)):
        if os.path.isdir(os.path.join(path,file)):
            process_audios(os.path.join(dir,file))
        elif file.split('.')[-1] in FILE_EXTENSIONS:
            transcript_file = file.split('.')[0] + '.cha'
            transcript_path = os.path.join(TRANSCRIPT_DIR,dir,transcript_file)
            if not os.path.exists(transcript_path):
                print('[ERROR] File not found', transcript_path)
                continue
            timestamps = extract_timestamps(transcript_path)
            if len(timestamps)==0:
                print('[ERROR] Transcript without timestamps', transcript_path)
                continue
            split_audio(dir,file, timestamps)

def download_audios(tb_cookie):
    # === Configuration ===
    BASE_URL = "https://media.talkbank.org/dementia"
    BASE_URL_2 = "https://media.talkbank.org:443/dementia/"
    OUTPUT_DIR = "/buckets/projects/ct_igc_phase3/phase3/talkbank_dementia"
    #TODO: TAKE OUT SESSION COOKIE
    SESSION_COOKIE = tb_cookie
    FILE_EXTENSIONS = {".zip", ".mp3", ".mp4", ".wav", ".cha", ".xml", ".txt", ".pdf", ".tgz", ".xlsx", ".csv", ".xls"}

    # === Requests session with headers and cookie ===
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://media.talkbank.org/",
    })
    session.cookies.set("talkbank", SESSION_COOKIE, domain="media.talkbank.org")

    def get_links(url):
        try:
            response = session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the first (or only) table
            table = soup.find("table")
            if not table:
                return []

            # Get all <a href=...> inside the table
            links = []
            for a in table.find_all("a", href=True):
                full_url = urljoin(url, a['href'])
                links.append(full_url)

            links = list(set([link.split('?')[0] for link in links]))
            return links

        except Exception as e:
            print(f"[ERROR] Cannot parse links from {url}: {e}")
            return []

    def download_file(url, base_url):

        rel_path = url[len(base_url):]
        local_path = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if not local_path.endswith('.mp3') and not local_path.endswith('.wav'):
            print('ZIP')

        # Skip already fully downloaded files
        if os.path.exists(local_path):
            print(f"[SKIP] Already exists: {rel_path}")
            return

        # HEAD request to get total file size
        head = session.head(url)
        if head.status_code != 206 and head.status_code != 200:
            print(f"[HEAD ERROR] {url}: {head.status_code}")
            return

        if head.status_code == 200:
            total_size = int(head.headers.get("Content-Length",""))
        else:
            total_size = int(head.headers.get("Content-Range", "").split('/')[-1])

        if not total_size:
            print(f"[ERROR] No valid Content-Range found for {url}")
            return

        chunk_size = 1024 * 1024  # 1MB
        print(f"[DOWNLOADING] {rel_path} ({total_size:,} bytes)")

        try:
            with open(local_path, "wb") as f:
                if head.status_code == 206:
                    for start in range(0, total_size, chunk_size):
                        end = min(start + chunk_size - 1, total_size - 1)
                        headers = {"Range": f"bytes={start}-{end}"}
                        r = session.get(url, headers=headers, stream=True)
                        if r.status_code != 206 and r.status_code != 200:
                            print(f"[ERROR] Bad status {r.status_code} for range {start}-{end}")
                            return
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    r = session.get(url, stream=True)
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"[DONE] {rel_path}")
        except Exception as e:
            print(f"[FAILED] {url}: {e}")

    def crawl(url):
        try:
            response = session.get(url, stream=True)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")

            # If it's a downloadable file
            if any(ext in url.lower() for ext in FILE_EXTENSIONS) or not content_type.startswith("text/html"):
                download_file(url, BASE_URL_2)
                return

            # Otherwise, it's a directory listing, parse links
            links = get_links(url)
            for link in links:
                # Prevent going up to parent directories or leaving base
                if link.startswith(BASE_URL_2):
                    crawl(link)

        except Exception as e:
            print(f"[ERROR] {url}: {e}")

    crawl(BASE_URL)

def download_transcripts(tb_cookie):
    # === Configuration ===
    BASE_URL = "https://git.talkbank.org/dementia/data-orig"
    BASE_URL_2 = "https://git.talkbank.org:443/dementia/data-orig/"
    OUTPUT_DIR = "/buckets/projects/ct_igc_phase3/phase3/talkbank_dementia_transcripts"
    #TODO: TAKE OUT SESSION COOKIE
    #TODO: EXPLAIN WHERE TO OBTAIN THIS COOKIE
    SESSION_COOKIE = tb_cookie
    FILE_EXTENSIONS = {".cha"}

    # === Requests session with headers and cookie ===
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "Referer": "https://git.talkbank.org/",
    })
    session.cookies.set("talkbank", SESSION_COOKIE, domain="git.talkbank.org")

    def is_file_link(href):
        return any(href.lower().endswith(ext) for ext in FILE_EXTENSIONS)

    def get_links(url):
        try:
            response = session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Find the first (or only) table
            table = soup.find("table")
            if not table:
                return []

            # Get all <a href=...> inside the table
            links = []
            for a in table.find_all("a", href=True):
                full_url = urljoin(url, a['href'])
                links.append(full_url)

            links = list(set([link.split('?')[0] for link in links]))
            return links

        except Exception as e:
            print(f"[ERROR] Cannot parse links from {url}: {e}")
            return []

    def download_file(url, base_url):
        rel_path = url[len(base_url):]
        local_path = os.path.join(OUTPUT_DIR, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Skip already fully downloaded files
        if os.path.exists(local_path):
            print(f"[SKIP] Already exists: {rel_path}")
            return

        # HEAD request to get total file size
        head = session.head(url)
        if head.status_code != 206 and head.status_code != 200:
            print(f"[HEAD ERROR] {url}: {head.status_code}")
            return

        if head.status_code == 200:
            total_size = int(head.headers.get("Content-Length",""))
        else:
            total_size = int(head.headers.get("Content-Range", "").split('/')[-1])

        if not total_size:
            print(f"[ERROR] No valid Content-Range found for {url}")
            return

        chunk_size = 1024 * 1024  # 1MB
        print(f"[DOWNLOADING] {rel_path} ({total_size:,} bytes)")

        try:
            with open(local_path, "wb") as f:
                if head.status_code == 206:
                    for start in range(0, total_size, chunk_size):
                        end = min(start + chunk_size - 1, total_size - 1)
                        headers = {"Range": f"bytes={start}-{end}"}
                        r = session.get(url, headers=headers, stream=True)
                        if r.status_code != 206 and r.status_code != 200:
                            print(f"[ERROR] Bad status {r.status_code} for range {start}-{end}")
                            return
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                else:
                    r = session.get(url, stream=True)
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            print(f"[DONE] {rel_path}")
        except Exception as e:
            print(f"[FAILED] {url}: {e}")

    def crawl(url):
        try:
            response = session.get(url, stream=True)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")

            # If it's a downloadable file
            if any(ext in url.lower() for ext in FILE_EXTENSIONS) or not content_type.startswith("text/html"):
                download_file(url, BASE_URL_2)
                return

            # Otherwise, it's a directory listing, parse links
            links = get_links(url)
            for link in links:
                # Prevent going up to parent directories or leaving base
                if link.startswith(BASE_URL_2):
                    crawl(link)

        except Exception as e:
            print(f"[ERROR] {url}: {e}")

    # if __name__ == "__main__":
    crawl(BASE_URL)

def process_holland_audios():
    base_path = '/buckets/projects/ct_igc_phase3/phase3/talkbank_dementia'

    path = os.path.join(base_path,'English/Holland')
    for file in os.listdir(path):
        if file.endswith('.mp4'):
            with VideoFileClip(os.path.join(path,file)) as video:
                audio = video.audio
                audio.write_audiofile(os.path.join(path,file.replace('mp4','mp3')))

    # Take segmented audios (extracting interviewer segments) and split by participant
    saving_path = os.path.join(base_path+'_processed','English/Holland')
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    participants = []
    audio_1 = AudioSegment.from_file(os.path.join(path,'tele01a.mp3'))
    audio_2 = AudioSegment.from_file(os.path.join(path,'tele01b.mp3'))
    audio_3 = AudioSegment.from_file(os.path.join(path,'tele01c.mp3'))

    audio_1[:51000].export(os.path.join(saving_path,'participant_1.mp3'))
    (audio_1[51000:]+audio_2[:75500]).export(os.path.join(saving_path,'participant_2.mp3'))
    audio_2[75500:].export(os.path.join(saving_path,'participant_3.mp3'))
    audio_3.export(os.path.join(saving_path,'participant_4.mp3'))

def main(tb_cookie):
    # Define the URL
    url = "https://media.talkbank.org/dementia"

    # Define headers to mimic the original request
    headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Cookie": f"talkbank={tb_cookie}",
    "Priority": "u=0, i",
    "Referer": "https://media.talkbank.org/",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15"
    }

    # Make the GET request
    response = requests.get(url, headers=headers)

    # Print the status code and content
    print(f"Status Code: {response.status_code}")
    with open("dementia_page.html", "w", encoding="utf-8") as file:
        file.write(response.text)

    print("HTML content saved to dementia_page.html")

    download_audios(tb_cookie)
    process_holland_audios()
    download_transcripts(tb_cookie)
    process_audios('')




    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process TalkBank dementia data.")
    parser.add_argument('--cookie', type=str, required=True, help='Session cookie for TalkBank')
    #TODO: ADD AN ARGS TO GET THE BASE PATH
    args = parser.parse_args()
    main(args.cookie)