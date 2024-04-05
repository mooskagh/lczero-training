package scanner

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

type FileContent struct {
	Filename string
	Content  []byte
}

type compressedFileContent struct {
	filename string
	content  []byte
}

type Scanner struct {
	output       chan FileContent
	gunzipOutput chan compressedFileContent
	input        chan string
	read_wg      sync.WaitGroup
	gunzip_wg    sync.WaitGroup
}

func (s *Scanner) processFile(filename string) error {
	fileInfo, err := os.Stat(filename)
	if err != nil {
		log.Println("Error getting file info:", err)
		return err
	}

	if fileInfo.IsDir() {
		return s.processDirectory(filename)
	}

	if strings.HasSuffix(filename, ".tar") {
		return s.processTarFile(filename)
	}

	if strings.HasSuffix(filename, ".gz") {
		return s.processGzippedFile(filename)
	}

	return s.processRegularFile(filename)
}

func (s *Scanner) processDirectory(filename string) error {
	files, err := os.ReadDir(filename)
	if err != nil {
		log.Println("Error reading directory:", err)
		return err
	}

	for _, f := range files {
		if !f.IsDir() {
			s.input <- filepath.Join(filename, f.Name())
		}
	}

	return nil
}

func (s *Scanner) processTarFile(filename string) error {
	tarFile, err := os.Open(filename)
	if err != nil {
		log.Println("Error opening tar file:", err)
		return err
	}
	defer tarFile.Close()

	tarReader := tar.NewReader(tarFile)
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Println("Error reading tar file:", err)
			return err
		}
		fileContent, err := io.ReadAll(tarReader)
		if err != nil {
			log.Println("Error reading tar entry:", err)
			return err
		}
		if strings.HasSuffix(header.Name, ".gz") {
			s.gunzipOutput <- compressedFileContent{
				filename: header.Name,
				content:  fileContent,
			}
		} else {
			s.output <- FileContent{
				Filename: header.Name,
				Content:  fileContent,
			}
		}
	}

	return nil
}

func (s *Scanner) processGzippedFile(filename string) error {
	fileContent, err := os.ReadFile(filename)
	if err != nil {
		log.Println("Error reading gzipped file:", err)
		return err
	}

	s.gunzipOutput <- compressedFileContent{
		filename: filename,
		content:  fileContent,
	}

	return nil
}

func (s *Scanner) processRegularFile(filename string) error {
	fileContent, err := os.ReadFile(filename)
	if err != nil {
		log.Println("Error reading file:", err)
		return err
	}

	s.output <- FileContent{
		Filename: filename,
		Content:  fileContent,
	}
	return nil
}

func (s *Scanner) gunzipWorker() {
	defer s.gunzip_wg.Done()

	for compressedFile := range s.gunzipOutput {
		reader, err := gzip.NewReader(bytes.NewReader(compressedFile.content))
		if err != nil {
			log.Println("Error creating gzip reader:", err)
			continue
		}

		fileContent, err := io.ReadAll(reader)
		if err != nil {
			log.Println("Error reading gunzipped file:", err)
			continue
		}

		reader.Close()

		s.output <- FileContent{
			Filename: strings.TrimSuffix(compressedFile.filename, ".gz"),
			Content:  fileContent,
		}
	}
}

func (s *Scanner) worker() {
	defer s.read_wg.Done()

	for filename := range s.input {
		s.processFile(filename)
	}
}

func (s *Scanner) GetOutput() <-chan FileContent {
	return s.output
}

func (s *Scanner) AddInput(filename string) {
	s.input <- filename
}

func (s *Scanner) Close() {
	close(s.input)
	s.read_wg.Wait()
	close(s.gunzipOutput)
	s.gunzip_wg.Wait()
	close(s.output)
}

func NewScanner(numWorkers int, numGunzipWorkers int) *Scanner {
	s := Scanner{
		output:       make(chan FileContent),
		gunzipOutput: make(chan compressedFileContent),
		input:        make(chan string),
	}
	// Start worker pool
	for i := 0; i < numWorkers; i++ {
		s.read_wg.Add(1)
		go s.worker()
	}

	// Start gunzip worker pool
	for i := 0; i < numGunzipWorkers; i++ {
		s.gunzip_wg.Add(1)
		go s.gunzipWorker()
	}

	return &s
}
