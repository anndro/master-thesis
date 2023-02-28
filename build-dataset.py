# -*- coding: utf-8 -*-
import os
import csv
import urllib2

def get_imdb(os_id, imdb_checked_list):
    
    req = urllib2.Request('https://www.opensubtitles.org/en/subtitles/' + os_id)
    
    try:
        response = urllib2.urlopen(req)
        osub_page = response.read()
    except:
        print("Error at" + os_id)
        osub_page = ''

    start_pos = osub_page.find('https://www.imdb.com/title/')
    end_pos = osub_page.find('/', start_pos+30)
    imdb_link = osub_page[start_pos:end_pos]

    if not imdb_link:
        return False, []

    if (imdb_link.replace('https://www.imdb.com/title/', '') in imdb_checked_list):
        return 'Duplicate', []

    imdb_req = urllib2.Request(imdb_link)
    try:
        imdb_response = urllib2.urlopen(imdb_req)
        imdb_page = imdb_response.read()
    except:
        print("Error at" + os_id)
        return False, []

    start_pos = imdb_page.find('<h4 class="inline">Genres:</h4>')
    end_pos = imdb_page.find('</div>', start_pos)
    genres = imdb_page[start_pos:end_pos]

    genre_list = genres.split('</a>')
    genre_clean_list = []

    for genre in genre_list:
        start = genre.find("> ")
        if start > 0:
            genre_clean_list.append(genre[start+2:])

    return imdb_link.replace('https://www.imdb.com/title/', ''), genre_clean_list


file_reader = open('id_match.csv', 'rb')
genre_reader = csv.reader(file_reader, delimiter=',', quotechar='|')
imdb_checked_list = []
osub_checked_list = []

for g in genre_reader:
    osub_checked_list.append(g[0])
    imdb_checked_list.append(g[1])

file_reader.close()

csvfile = open('id_match.csv', 'ab')
genre_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

for root, dirs, files in os.walk("raw"):

    find_os = 0

    for file in files:
        if file.find('.xml.gz') > 0:
            os_id = file.replace('.xml.gz', '')
            if os_id in osub_checked_list:
                print('%s already checked' % os_id)
            elif find_os:
                print('%s same folder' % os_id)
            else:
                imdb_id, genres = get_imdb(os_id, imdb_checked_list)
                imdb_checked_list.append(imdb_id)
                print('- %s -> %s printed' % (os_id, imdb_id))
                genre_writer.writerow([os_id, imdb_id, ','.join(genres)])
                find_os = 1

