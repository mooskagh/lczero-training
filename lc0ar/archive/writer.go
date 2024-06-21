//go:generate protoc --go_out=. --go_opt=paths=source_relative lc0ar.proto
package archive

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	sync "sync"

	"github.com/DataDog/zstd"
	"google.golang.org/protobuf/proto"
)

var (
	magic   string = "lc0ar"
	Version uint32 = 1
)

type FileContent struct {
	Filename string
	Content  []byte
}

type block struct {
	header  TrainingDataArchive
	payload bytes.Buffer
}

type archiveFileWriter struct {
	file     *zstd.Writer
	pageSize uint32
	block    *block
}

func divideRoundUp(a, b uint32) uint32 {
	return (a + b - 1) / b
}

func transpose(m uint32, src []byte) []byte {
	const tileSize = 256
	const numWorkers = 10
	dst := make([]byte, len(src))

	// Transposing N×M into M×N.
	n := uint32(len(src)) / m
	m_tiles := divideRoundUp(m, tileSize)
	n_tiles := divideRoundUp(n, tileSize)

	tasks := make(chan uint32)
	var wg sync.WaitGroup

	worker := func() {
		defer wg.Done()

		for tile_id := range tasks {
			var i_tile = tile_id / m_tiles
			var j_tile = tile_id % m_tiles
			var i_limit = min(tileSize, n-i_tile*tileSize)
			var j_limit = min(tileSize, m-j_tile*tileSize)
			for ii := uint32(0); ii < i_limit; ii++ {
				for jj := uint32(0); jj < j_limit; jj++ {
					dst[(j_tile*tileSize+jj)*n+i_tile*tileSize+ii] =
						src[(i_tile*tileSize+ii)*m+j_tile*tileSize+jj]
				}
			}
		}
	}

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker()
	}

	total_tiles := m_tiles * n_tiles
	for i := uint32(0); i < total_tiles; i++ {
		tasks <- i
	}
	close(tasks)
	wg.Wait() // Wait for all goroutines to finish

	return dst
}

func (w *archiveFileWriter) writeBlock(block *block) error {
	src := block.payload.Bytes()
	dst := transpose(w.pageSize, src)

	headerBytes, err := proto.Marshal(&block.header)
	if err != nil {
		return fmt.Errorf("failed to marshal header: %w", err)
	}
	headerLength := uint32(len(headerBytes))
	if err := binary.Write(w.file, binary.BigEndian, headerLength); err != nil {
		return fmt.Errorf("failed to write header length: %w", err)
	}
	if _, err := (*w.file).Write(headerBytes); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}
	if _, err := (*w.file).Write(dst); err != nil {
		return fmt.Errorf("failed to write payload: %w", err)
	}
	// w.file.Flush()
	return nil
}

func newBlock() *block {
	return &block{
		header: TrainingDataArchive{
			Magic:   &magic,
			Version: &Version,
		},
		payload: bytes.Buffer{},
	}
}

func (w *archiveFileWriter) flush() error {
	block_to_write := w.block
	w.block = newBlock()
	if err := w.writeBlock(block_to_write); err != nil {
		fmt.Printf("Error writing block: %v\n", err)
		return err
	}
	return nil
}

func (w *archiveFileWriter) Write(fc FileContent) error {
	// Append the file content to the buffer, and pad to the multiple of frameSize.
	w.block.payload.Write(fc.Content)
	partialFrame := len(fc.Content) % int(w.pageSize)
	if partialFrame != 0 {
		padding := make([]byte, int(w.pageSize)-partialFrame)
		w.block.payload.Write(padding)
	}
	fileLength := uint32(len(fc.Content))
	w.block.header.FileMetadata = append(w.block.header.FileMetadata, &FileMetadata{
		Name:      &fc.Filename,
		SizeBytes: &fileLength,
	})

	if w.block.payload.Len() >= 512*1024*1024 {
		if err := w.flush(); err != nil {
			return err
		}
	}
	return nil
}

func NewArchiveFileWriter(
	filename string, append bool, pageSize uint32) (*archiveFileWriter, error) {
	var flags int
	if append {
		flags = os.O_APPEND | os.O_CREATE | os.O_WRONLY
	} else {
		flags = os.O_TRUNC | os.O_CREATE | os.O_WRONLY
	}

	file, err := os.OpenFile(filename, flags, 0644)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}

	zWriter := zstd.NewWriter(file)

	w := archiveFileWriter{
		file:     zWriter,
		pageSize: pageSize,
		block:    newBlock(),
	}
	return &w, nil
}

func (w *archiveFileWriter) Close() error {
	if w.block.payload.Len() >= 0 {
		if err := w.flush(); err != nil {
			return err
		}
	}
	return w.file.Close()
}
