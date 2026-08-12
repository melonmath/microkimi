// Generic mini JSON parser: null, bool, number, string, array, object.
// Sufficient for safetensors headers, index.json and golden.json.
// Not a general JSON library: whole document parsed in memory, no
// streaming, no validation beyond the grammar.

#[derive(Debug, Clone)]
pub enum Json {
    Null,
    #[allow(dead_code)]
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        if let Json::Obj(pairs) = self {
            pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v)
        } else {
            None
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self {
            Some(s)
        } else {
            None
        }
    }
    pub fn as_num(&self) -> Option<f64> {
        if let Json::Num(n) = self {
            Some(*n)
        } else {
            None
        }
    }
    #[allow(dead_code)]
    pub fn as_arr(&self) -> Option<&Vec<Json>> {
        if let Json::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }
}

pub fn parse(bytes: &[u8]) -> Json {
    let mut p = Parser { b: bytes, i: 0 };
    p.value()
}

/// Parses one complete JSON value and rejects non-whitespace trailing bytes.
/// Strict bounded-format readers use this when the exact byte range matters.
pub fn parse_complete(bytes: &[u8]) -> Json {
    let mut p = Parser { b: bytes, i: 0 };
    let value = p.value();
    p.ws();
    assert_eq!(p.i, bytes.len(), "JSON: trailing bytes at {}", p.i);
    value
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn peek(&self) -> u8 {
        if self.i < self.b.len() {
            self.b[self.i]
        } else {
            0
        }
    }
    fn expect(&mut self, c: u8) {
        assert!(self.peek() == c, "JSON: expected '{}' at {}", c as char, self.i);
        self.i += 1;
    }
    fn value(&mut self) -> Json {
        self.ws();
        match self.peek() {
            b'{' => self.obj(),
            b'[' => self.arr(),
            b'"' => Json::Str(self.string()),
            b't' => {
                self.i += 4;
                Json::Bool(true)
            }
            b'f' => {
                self.i += 5;
                Json::Bool(false)
            }
            b'n' => {
                self.i += 4;
                Json::Null
            }
            _ => self.num(),
        }
    }
    fn obj(&mut self) -> Json {
        self.expect(b'{');
        let mut out = Vec::new();
        self.ws();
        if self.peek() == b'}' {
            self.i += 1;
            return Json::Obj(out);
        }
        loop {
            self.ws();
            let key = self.string();
            self.ws();
            self.expect(b':');
            let val = self.value();
            out.push((key, val));
            self.ws();
            match self.peek() {
                b',' => {
                    self.i += 1;
                }
                b'}' => {
                    self.i += 1;
                    break;
                }
                c => panic!("JSON: unexpected '{}' at {}", c as char, self.i),
            }
        }
        Json::Obj(out)
    }
    fn arr(&mut self) -> Json {
        self.expect(b'[');
        let mut out = Vec::new();
        self.ws();
        if self.peek() == b']' {
            self.i += 1;
            return Json::Arr(out);
        }
        loop {
            out.push(self.value());
            self.ws();
            match self.peek() {
                b',' => {
                    self.i += 1;
                }
                b']' => {
                    self.i += 1;
                    break;
                }
                c => panic!("JSON: unexpected '{}' at {}", c as char, self.i),
            }
        }
        Json::Arr(out)
    }
    fn num(&mut self) -> Json {
        let start = self.i;
        while self.i < self.b.len()
            && matches!(self.b[self.i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        {
            self.i += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.i]).unwrap();
        Json::Num(s.parse::<f64>().unwrap_or(0.0))
    }
    fn hex4(&mut self) -> u32 {
        let s = std::str::from_utf8(&self.b[self.i..self.i + 4]).unwrap();
        self.i += 4;
        u32::from_str_radix(s, 16).unwrap()
    }
    fn string(&mut self) -> String {
        self.expect(b'"');
        let mut out = String::new();
        loop {
            let c = self.b[self.i];
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.b[self.i];
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4();
                            if (0xD800..0xDC00).contains(&hi) {
                                assert_eq!(&self.b[self.i..self.i + 2], b"\\u");
                                self.i += 2;
                                let lo = self.hex4();
                                let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                                out.push(char::from_u32(cp).unwrap());
                            } else {
                                out.push(char::from_u32(hi).unwrap());
                            }
                        }
                        _ => panic!("JSON: unknown escape \\{}", e as char),
                    }
                }
                _ => {
                    let start = self.i - 1;
                    let len = if c < 0x80 {
                        1
                    } else if c < 0xE0 {
                        2
                    } else if c < 0xF0 {
                        3
                    } else {
                        4
                    };
                    self.i = start + len;
                    out.push_str(std::str::from_utf8(&self.b[start..self.i]).unwrap());
                }
            }
        }
        out
    }
}
