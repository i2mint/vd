# vd.text

Text preprocessing and chunking utilities for vd.

Provides functions to clean, normalize, and chunk text before adding to
vector databases.

### Functions

| [`chunk_documents`](#vd.text.chunk_documents)(documents[, chunk_size, ...])    | Chunk multiple documents while preserving metadata.   |
|---------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| [`chunk_text`](#vd.text.chunk_text)(text[, chunk_size, overlap, ...])     | Chunk text into smaller pieces.                       |
| [`clean_text`](#vd.text.clean_text)(text, \*[, lowercase, ...])           | Clean and normalize text.                             |
| [`extract_metadata`](#vd.text.extract_metadata)(text, \*[, extract_title, ...]) | Extract metadata from text.                           |
| [`normalize_whitespace`](#vd.text.normalize_whitespace)(text)                       | Normalize whitespace in text.                         |
| [`truncate_text`](#vd.text.truncate_text)(text, max_length, \*[, suffix])    | Truncate text to maximum length.                      |

### vd.text.chunk_documents(documents, chunk_size=500, , overlap=50, strategy='chars', id_template='{doc_id}_chunk_{chunk_num}', preserve_metadata=True)

Chunk multiple documents while preserving metadata.

* **Parameters:**
  * **documents** ([`Iterator`](https://docs.python.org/3/library/typing.html#typing.Iterator)[[`tuple`](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`str`](https://docs.python.org/3/builtins/stdtypes.html#str) | [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)]]) – Iterator of (doc_id, text) or (doc_id, text, metadata) tuples
  * **chunk_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Size of each chunk
  * **overlap** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Overlap between chunks
  * **strategy** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Chunking strategy (see chunk_text)
  * **id_template** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Template for chunk IDs. Can use {doc_id} and {chunk_num}
  * **preserve_metadata** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Whether to copy metadata to all chunks
* **Yields:**
  *tuple* – (chunk_id, chunk_text, metadata) tuples
* **Return type:**
  [*Iterator*](https://docs.python.org/3/library/typing.html#typing.Iterator)[[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*str*](https://docs.python.org/3/builtins/stdtypes.html#str), [*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)]]

### Examples

```pycon
>>> docs = [('doc1', 'Long text...', {'author': 'Alice'})]
>>> chunks = list(chunk_documents(docs, chunk_size=20))
>>> len(chunks) >= 1
True
```

### vd.text.chunk_text(text, chunk_size=500, , overlap=50, strategy='chars', preserve_sentences=True)

Chunk text into smaller pieces.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to chunk
  * **chunk_size** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Target size of each chunk (in characters or tokens depending on strategy)
  * **overlap** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Number of characters/tokens to overlap between chunks
  * **strategy** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – 

    Chunking strategy:
    - ’chars’: Character-based chunking
    - ’words’: Word-based chunking
    - ’sentences’: Sentence-based chunking
    - ’paragraphs’: Paragraph-based chunking
  * **preserve_sentences** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Try to avoid breaking sentences when using chars/words strategy
* **Returns:**
  List of text chunks
* **Return type:**
  [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str)]

### Examples

```pycon
>>> text = "This is sentence one. This is sentence two. This is sentence three."
>>> chunks = chunk_text(text, chunk_size=30, strategy='chars')
>>> len(chunks) >= 2
True
```

```pycon
>>> chunks = chunk_text(text, strategy='sentences')
>>> len(chunks)
3
```

### vd.text.clean_text(text, , lowercase=False, remove_extra_whitespace=True, remove_urls=False, remove_emails=False, remove_numbers=False, remove_punctuation=False)

Clean and normalize text.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to clean
  * **lowercase** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Convert to lowercase
  * **remove_extra_whitespace** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Collapse multiple spaces/newlines
  * **remove_urls** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Remove URLs
  * **remove_emails** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Remove email addresses
  * **remove_numbers** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Remove numbers
  * **remove_punctuation** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Remove punctuation
* **Returns:**
  Cleaned text
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> text = "Hello   World!  Visit https://example.com"
>>> clean_text(text, remove_urls=True)
'Hello World! Visit'
>>> clean_text(text, lowercase=True, remove_punctuation=True)
'hello world visit httpsexamplecom'
```

### vd.text.extract_metadata(text, , extract_title=True, extract_length=True, extract_word_count=True, extract_language=False)

Extract metadata from text.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to analyze
  * **extract_title** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Extract first line as title
  * **extract_length** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Add text length
  * **extract_word_count** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Add word count
  * **extract_language** ([`bool`](https://docs.python.org/3/builtins/functions.html#bool)) – Detect language (requires langdetect)
* **Returns:**
  Extracted metadata
* **Return type:**
  [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)[[`str`](https://docs.python.org/3/builtins/stdtypes.html#str), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)]

### Examples

```pycon
>>> text = "My Title\n\nThis is the content."
>>> meta = extract_metadata(text)
>>> meta['title']
'My Title'
>>> meta['char_count']
30
```

### vd.text.normalize_whitespace(text)

Normalize whitespace in text.

Replaces tabs, multiple spaces, and multiple newlines with single versions.

* **Parameters:**
  **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to normalize
* **Returns:**
  Normalized text
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> normalize_whitespace("Hello\t\tWorld  \n\n\nTest")
'Hello World \nTest'
```

### vd.text.truncate_text(text, max_length, , suffix='...')

Truncate text to maximum length.

* **Parameters:**
  * **text** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Text to truncate
  * **max_length** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – Maximum length
  * **suffix** ([`str`](https://docs.python.org/3/builtins/stdtypes.html#str)) – Suffix to add to truncated text
* **Returns:**
  Truncated text
* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### Examples

```pycon
>>> truncate_text("This is a long text", 10)
'This is...'
```
