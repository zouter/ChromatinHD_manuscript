import pysam

# rs443623
# found: A = ref
samfile = pysam.AlignmentFile("atac_possorted_bam.bam", "rb")
pileup = samfile.pileup("chr6", 33028572, 33028573)

for pileupcolumn in pileup:
    if pileupcolumn.pos in [33028571, 33028572, 33028573]:
        print("\ncoverage at base %s = %s" % (pileupcolumn.pos, pileupcolumn.n))
        for pileupread in pileupcolumn.pileups:
            if not pileupread.is_del and not pileupread.is_refskip:
                # query position is None if is_del or is_refskip is set.
                print(
                    "\tbase in read %s = %s"
                    % (pileupread.alignment.query_name, pileupread.alignment.query_sequence[pileupread.query_position])
                )


# rs443623
# found: A/C HETEROZYGOUS
samfile = pysam.AlignmentFile("atac_possorted_bam.bam", "rb")
pileup = samfile.pileup("chr1", 24970251, 24970252)

for pileupcolumn in pileup:
    if pileupcolumn.pos in [24970250, 24970251, 24970252]:
        print("\ncoverage at base %s = %s" % (pileupcolumn.pos, pileupcolumn.n))
        for pileupread in pileupcolumn.pileups:
            if not pileupread.is_del and not pileupread.is_refskip:
                # query position is None if is_del or is_refskip is set.
                print(
                    "\tbase in read %s = %s"
                    % (pileupread.alignment.query_name, pileupread.alignment.query_sequence[pileupread.query_position])
                )


# rs7668673
# found: G/C HETEROZYGOUS
samfile = pysam.AlignmentFile("atac_possorted_bam.bam", "rb")
pileup = samfile.pileup("chr4", 6917946, 6917947)

for pileupcolumn in pileup:
    if pileupcolumn.pos in [6917945, 6917946, 6917947]:
        print("\ncoverage at base %s = %s" % (pileupcolumn.pos, pileupcolumn.n))
        for pileupread in pileupcolumn.pileups:
            if not pileupread.is_del and not pileupread.is_refskip:
                # query position is None if is_del or is_refskip is set.
                print(
                    "\tbase in read %s = %s"
                    % (pileupread.alignment.query_name, pileupread.alignment.query_sequence[pileupread.query_position])
                )


# rs875742
# found: G/C HETEROZYGOUS
samfile = pysam.AlignmentFile("atac_possorted_bam.bam", "rb")
pileup = samfile.pileup("chr5", 173860759, 173860760)

for pileupcolumn in pileup:
    if pileupcolumn.pos in [173860758, 173860759, 173860760]:
        print("\ncoverage at base %s = %s" % (pileupcolumn.pos, pileupcolumn.n))
        for pileupread in pileupcolumn.pileups:
            if not pileupread.is_del and not pileupread.is_refskip:
                # query position is None if is_del or is_refskip is set.
                print(
                    "\tbase in read %s = %s"
                    % (pileupread.alignment.query_name, pileupread.alignment.query_sequence[pileupread.query_position])
                )
