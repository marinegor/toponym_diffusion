import polars as pl
from textwrap import dedent


class GeonamesDataset:
    _annotation = dedent(r"""
    geonameid         : integer id of record in geonames database
    name              : name of geographical point (utf8) varchar(200)
    asciiname         : name of geographical point in plain ascii characters, varchar(200)
    alternatenames    : alternatenames, comma separated, ascii names automatically transliterated, convenience attribute from alternatename table, varchar(10000)
    latitude          : latitude in decimal degrees (wgs84)
    longitude         : longitude in decimal degrees (wgs84)
    feature class     : see http://www.geonames.org/export/codes.html, char(1)
    feature code      : see http://www.geonames.org/export/codes.html, varchar(10)
    country code      : ISO-3166 2-letter country code, 2 characters
    cc2               : alternate country codes, comma separated, ISO-3166 2-letter country code, 200 characters
    admin1 code       : fipscode (subject to change to iso code), see exceptions below, see file admin1Codes.txt for display names of this code; varchar(20)
    admin2 code       : code for the second administrative division, a county in the US, see file admin2Codes.txt; varchar(80) 
    admin3 code       : code for third level administrative division, varchar(20)
    admin4 code       : code for fourth level administrative division, varchar(20)
    population        : bigint (8 byte int) 
    elevation         : in meters, integer
    dem               : digital elevation model, srtm3 or gtopo30, average elevation of 3''x3'' (ca 90mx90m) or 30''x30'' (ca 900mx900m) area in meters, integer. srtm processed by cgiar/ciat.
    timezone          : the iana timezone id (see file timeZone.txt) varchar(40)
    modification date : date of last modification in yyyy-MM-dd format""")

    def __init__(self, path: str, max_len: int = 6, min_len: int = 15):
        self.path = path
        self._colnames = list(
            map(lambda s: s.split(":")[0].strip(), self._annotation.split("\n"))
        )
        self._colnames = [c for c in self._colnames if c]
        self.read()
        self.process()

    def read(self) -> pl.DataFrame:
        self._raw_data = pl.read_csv(
            self.path,
            separator="\t",
            infer_schema_length=None,
            has_header=False,
            new_columns=self._colnames,
        )
        return self.raw_data

    def process(self):
        self._df = (
            self.raw_data.filter(
                pl.col("asciiname").str.replace_all("[^\p{Ascii}]", "")
                == pl.col(
                    "asciiname"
                ),  # remove all rows that contain non-ascii symbols
                pl.col("asciiname").str.len_chars()
                >= self.min_len,  # remove too short names
                pl.col("asciiname").str.len_chars()
                <= self.max_len,  # remove too long names
                pl.col("asciiname").str.contains("[[:^alpha:]]").not_(),
            )
            .select(["asciiname", "feature code", "country code", "population"])
            .rename({"asciiname": "sequence"})
            .filter(pl.all_horizontal(pl.col("*").is_not_null()))
        )

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def df(self):
        return self._df
