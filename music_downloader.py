from __future__ import unicode_literals
import os
import codecs
import youtube_dl

if __name__ == '__main__':

    class MyLogger(object):
        def debug(self, msg):
            pass

        def warning(self, msg):
            pass

        def error(self, msg):
            print(msg)


    def my_hook(d):
        if d['status'] == 'finished':
            print('Done downloading, now converting ...')

    category = ""
    ydl_opts = {}

    with codecs.open("music_list_p.txt", "r", "utf-8") as txtfile:
        raw = txtfile.readlines()
        for dat in raw:
            url = ""
            dat_split = dat.split(",")
            print(dat_split)
            if dat_split[0] != '':
                jan = dat_split[0]
                print(jan)
                category = "./" + "music/" + jan + "/"
                print(category)

                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': category + '%(title)s.%(ext)s',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'logger': MyLogger(),
                    'progress_hooks': [my_hook],
                }
                continue
            else:
                for u in dat_split:
                    if u[:5] == "https":
                        url = u
                print(url)

                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                #print(os.system("youtube-dl --extract-audio --audio-format mp3 " + dat_split[3]))
