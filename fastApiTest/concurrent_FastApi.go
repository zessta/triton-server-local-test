package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

var url = "http://127.0.0.1:7000/infer"
var concurrentRequests = []int{125, 130, 145}

type ResponseData struct {
	StatusCode        int
	OutputShape       string
	InferenceTimeSec  float64
	ConcurrentRequest int
}

// Struct to match expected JSON response
type InferResponse struct {
	OutputShape interface{} `json:"output_shape"`
}

func hitModel() ResponseData {
	start := time.Now()
	resp, err := http.Get(url)
	if err != nil {
		fmt.Println("Request error:", err)
		return ResponseData{StatusCode: 0, OutputShape: "error", InferenceTimeSec: 0}
	}
	defer resp.Body.Close()

	var inferResp InferResponse
	_ = json.NewDecoder(resp.Body).Decode(&inferResp)

	duration := time.Since(start).Seconds()
	return ResponseData{
		StatusCode:       resp.StatusCode,
		OutputShape:      fmt.Sprintf("%v", inferResp.OutputShape),
		InferenceTimeSec: duration,
	}
}

func main() {
	var allResults []ResponseData

	for _, n := range concurrentRequests {
		fmt.Printf("\nHitting %d concurrent requests:\n", n)
		results := make([]ResponseData, n)
		var wg sync.WaitGroup
		wg.Add(n)

		for i := 0; i < n; i++ {
			go func(i int) {
				defer wg.Done()
				results[i] = hitModel()
				results[i].ConcurrentRequest = n
			}(i)
		}
		wg.Wait()

		allResults = append(allResults, results...)

		for i, r := range results {
			fmt.Printf("Request %d: time=%.4fs, output_shape=%s\n", i+1, r.InferenceTimeSec, r.OutputShape)
		}
	}

	// Save to CSV
	file, err := os.Create("inference_results.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Header
	writer.Write([]string{"status_code", "output_shape", "inference_time_sec", "concurrent_requests"})

	for _, r := range allResults {
		writer.Write([]string{
			strconv.Itoa(r.StatusCode),
			r.OutputShape,
			fmt.Sprintf("%.6f", r.InferenceTimeSec),
			strconv.Itoa(r.ConcurrentRequest),
		})
	}

	fmt.Println("\nResults saved to inference_results.csv")
}
