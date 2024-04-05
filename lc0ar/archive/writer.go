package archive

import (
	"fmt"
	"os"

	"kythe.io/kythe/go/util/riegeli"
)

type archiveFileWriter struct {
	file      *riegeli.Writer
	frameSize uint32
}

func NewArchiveFileWriter(
	filename string, append bool, frameSize uint32) (*archiveFileWriter, error) {
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

	rWriter := riegeli.NewWriter(file, &riegeli.WriterOptions{})

	w := archiveFileWriter{
		file:      rWriter,
		frameSize: frameSize,
	}
	return &w, nil
}

func (w *archiveFileWriter) Close() error {
	return w.file.Close()
}
