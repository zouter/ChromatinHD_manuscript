wget https://github.com/samtools/bcftools/releases/download/1.16/bcftools-1.16.tar.bz2
tar xjvf bcftools-1.16.tar.bz2


mkdir -p $HOME/bin $HOME/GRCh38 && cd /tmp

wget https://github.com/samtools/bcftools/releases/download/1.16/bcftools-1.16.tar.bz2
tar xjvf bcftools-1.16.tar.bz2

cd bcftools-1.16/
/bin/rm -f plugins/{gtc2vcf.{c,h},affy2vcf.c}
wget -P plugins https://raw.githubusercontent.com/freeseek/gtc2vcf/master/{gtc2vcf.{c,h},affy2vcf.c}
make
/bin/cp bcftools plugins/{gtc,affy}2vcf.so $HOME/bin/


wget -O- ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz | gzip -d > $HOME/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna
samtools faidx $HOME/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna
bwa index $HOME/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna


mkdir -p $HOME/bin && cd /tmp
wget ftp://webdata2:webdata2@ussd-ftp.illumina.com/downloads/software/iaap/iaap-cli-linux-x64-1.1.0.tar.gz
tar xzvf iaap-cli-linux-x64-1.1.0.tar.gz -C $HOME/bin/ iaap-cli-linux-x64-1.1.0/iaap-cli --strip-components=1





# minimac4
git clone https://github.com/statgen/Minimac4.git
cd Minimac4
bash install.sh
cp release-build/minimac4 $HOME/bin/



