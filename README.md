# alano

#### Fast zip and unzip using pigz
tar -c --use-compress-program=pigz -f tar.file dir_to_zip \
pigz -dc tar.file | pv | tar xf -
