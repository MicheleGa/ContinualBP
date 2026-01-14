#Pulse DB

## Download

# MIMIC III
curl -L -o "PulseDB_MIMIC.zip.001" -C - "https://rutgers.box.com/shared/static/7l8n3tn9tr0602tdss1x7e3uliahlibp.001"
curl -L -o "PulseDB_MIMIC.zip.002" -C - "https://rutgers.box.com/shared/static/zco48rvz5dog72970679foen6hct15c8.002"
curl -L -o "PulseDB_MIMIC.zip.003" -C - "https://rutgers.box.com/shared/static/x22qpmelx6sz3wgkm5qyc0eis429361f.003"
curl -L -o "PulseDB_MIMIC.zip.004" -C - "https://rutgers.box.com/shared/static/xj25sqnluiz6s4z8tzzm5phk00ohp6e8.004"
curl -L -o "PulseDB_MIMIC.zip.005" -C - "https://rutgers.box.com/shared/static/dxus2lsoop02chaspnwipwrf0g4wmenr.005"
curl -L -o "PulseDB_MIMIC.zip.006" -C - "https://rutgers.box.com/shared/static/rts6sj441laenm2sy1qcemg7ke4om3j6.006"
curl -L -o "PulseDB_MIMIC.zip.007" -C - "https://rutgers.box.com/shared/static/vor4hjllld7a0c3nzef8uptbb4ut3koo.007"
curl -L -o "PulseDB_MIMIC.zip.008" -C - "https://rutgers.box.com/shared/static/a2qg2p4ebyrooji3z88djlokji65tlf3.008"
curl -L -o "PulseDB_MIMIC.zip.009" -C - "https://rutgers.box.com/shared/static/uh6kbiuqgnib5wakiv6o35gkpusyamc7.009"
curl -L -o "PulseDB_MIMIC.zip.010" -C - "https://rutgers.box.com/shared/static/h6eyhkkx48pf3ce3th1clwj43hn98j5c.010"
curl -L -o "PulseDB_MIMIC.zip.011" -C - "https://rutgers.box.com/shared/static/e93dp94hxpkas45yc59n289s2wvkafgi.011"
curl -L -o "PulseDB_MIMIC.zip.012" -C - "https://rutgers.box.com/shared/static/iuvyuw7dmlxvbjvt53dj49wqn3gelqni.012"
curl -L -o "PulseDB_MIMIC.zip.013" -C - "https://rutgers.box.com/shared/static/qxx6tjz8c3778601ib3icu6o1rranmc7.013"
curl -L -o "PulseDB_MIMIC.zip.014" -C - "https://rutgers.box.com/shared/static/ip2ninwqj8437l9fyffjprnk90ptnx9k.014"
curl -L -o "PulseDB_MIMIC.zip.015" -C - "https://rutgers.box.com/shared/static/yrtbo0lg8mjhaw624iw9bbhk1obbocwd.015"
curl -L -o "PulseDB_MIMIC.zip.016" -C - "https://rutgers.box.com/shared/static/wmzndowgfa5xi3tvtqahxkld3ngdyjds.016"

# Vital DB
curl -L -o "PulseDB_Vital.zip.001" -C - "https://rutgers.box.com/shared/static/vtxoksmn7emeaxypb2prywgwscuefoqa.001"
curl -L -o "PulseDB_Vital.zip.002" -C - "https://rutgers.box.com/shared/static/euzkek7c3xoy62jisheuxqar7z5y8xig.002"
curl -L -o "PulseDB_Vital.zip.003" -C - "https://rutgers.box.com/shared/static/49lngo0benxfjw193jnqz9tctlyb3qam.003"
curl -L -o "PulseDB_Vital.zip.004" -C - "https://rutgers.box.com/shared/static/jf4fwgkmhry20mf5tcg9t0wxvky64um0.004"
curl -L -o "PulseDB_Vital.zip.005" -C - "https://rutgers.box.com/shared/static/2lgxysbskfuapsaan4jypvmm8316fdkc.005"
curl -L -o "PulseDB_Vital.zip.006" -C - "https://rutgers.box.com/shared/static/x27ktb4qsx43razwo4tjmxq9v1ro0x3y.006"
curl -L -o "PulseDB_Vital.zip.007" -C - "https://rutgers.box.com/shared/static/q0t36fikgf3pimhvnerwwnovfr0umtp8.007"
curl -L -o "PulseDB_Vital.zip.008" -C - "https://rutgers.box.com/shared/static/ihckx2g0f981g5yz2x8v5rgwndl6yebw.008"
curl -L -o "PulseDB_Vital.zip.009" -C - "https://rutgers.box.com/shared/static/y8j14h8tvi5b3du8nap9dnura1omfrk6.009"
curl -L -o "PulseDB_Vital.zip.010" -C - "https://rutgers.box.com/shared/static/fu0m9tx33jkxywq32shh0g8dg3not15u.010"

# To check downlaoded data integrity, assuming .sha1 file is present in the current directory: 
sha1sum -c checksum.sha1

## Unzip

# To list the files in the archive: 7zz l first_archive_part.zip.001
# To unzip the files in the archive: 7zz x first_archive_part.zip.001
7zz x PulseDB_MIMIC.zip.001
7zz x PulseDB_Vital.zip.001