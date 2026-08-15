VALID_BASES = set("ACGT")


def clean_sequence(sequence):
    return "".join(str(sequence).split()).upper()


def validate_sequences(records, expected_length):
    for record in records:
        sequence = record["sequence"]
        if len(sequence) != expected_length:
            raise ValueError(
                f"{record['id']}: expected {expected_length} bp, observed {len(sequence)} bp"
            )
        invalid = set(sequence) - VALID_BASES
        if invalid:
            raise ValueError(f"{record['id']}: unsupported bases: {''.join(sorted(invalid))}")


def overlapping_kmer(sequence, k=6):
    return " ".join(sequence[index : index + k] for index in range(len(sequence) - k + 1))


def prepare_sequences(records, mode):
    sequences = [record["sequence"] for record in records]
    if mode == "kmer6":
        return [overlapping_kmer(sequence) for sequence in sequences]
    return sequences
