import csv

from .preprocessing import clean_sequence


FASTA_SUFFIXES = {".fa", ".fasta", ".fna", ".fas"}
TABLE_SUFFIXES = {".csv", ".tsv", ".txt"}


def read_fasta(path):
    records = []
    record_id = None
    chunks = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if record_id is not None:
                    records.append({"id": record_id, "sequence": clean_sequence("".join(chunks))})
                record_id = line[1:].strip().split()[0]
                chunks = []
            else:
                if record_id is None:
                    raise ValueError("FASTA sequence appears before the first header")
                chunks.append(line)
    if record_id is not None:
        records.append({"id": record_id, "sequence": clean_sequence("".join(chunks))})
    return records


def read_table(path, sequence_column, id_column):
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames or sequence_column not in reader.fieldnames:
            raise ValueError(f"missing sequence column: {sequence_column}")
        records = []
        for index, row in enumerate(reader, start=1):
            record = dict(row)
            record["id"] = row.get(id_column) or f"sequence_{index}"
            record["sequence"] = clean_sequence(row[sequence_column])
            records.append(record)
    return records


def read_records(path, sequence_column, id_column):
    suffix = path.suffix.lower()
    if suffix in FASTA_SUFFIXES:
        return read_fasta(path)
    if suffix in TABLE_SUFFIXES:
        return read_table(path, sequence_column, id_column)
    raise ValueError("input must be FASTA, CSV, or TSV")


def write_tsv(path, records, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or [])
    if not fields:
        for record in records:
            for field in record:
                if field not in fields:
                    fields.append(field)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)
