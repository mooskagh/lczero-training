package main

import (
	"fmt"
	"sync/atomic"

	"github.com/LeelaChessZero/lc0ar/archive"
	"github.com/LeelaChessZero/lc0ar/scanner"
	"github.com/spf13/cobra"
)

func main() {
	var structSize int

	var addCmd = &cobra.Command{
		Use:   "add [archive name] [tar files or directories...]",
		Short: "Add training games to an archive",
		Args:  cobra.MinimumNArgs(2),
		Run: func(cmd *cobra.Command, args []string) {
			scanner := scanner.NewScanner(10, 5)
			fileStream := scanner.GetOutput()
			defer scanner.Close()

			var count int64 = 0

			go func() {
				writer, err := archive.NewArchiveFileWriter(args[0], false, uint32(structSize))
				if err != nil {
					fmt.Println("Error creating archive file:", err)
					return
				}
				defer writer.Close()

				for {
					fc := <-fileStream
					fmt.Printf("[%d] %s (%d)\n", count, fc.Filename, len(fc.Content))
					atomic.AddInt64(&count, 1)

					err := writer.Write(archive.FileContent{
						Filename: fc.Filename,
						Content:  fc.Content,
					})
					if err != nil {
						fmt.Println("Error writing file to archive:", err)
						return
					}
				}
			}()

			for i := 1; i < len(args); i++ {
				scanner.AddInput(args[i])
			}
		},
	}
	addCmd.Flags().IntVarP(&structSize, "struct-size", "s", 8356, "Size of the struct to use")

	var rootCmd = &cobra.Command{Use: "lc0ar"}
	rootCmd.AddCommand(addCmd)
	rootCmd.Execute()

}
