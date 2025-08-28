package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

var url = "http://127.0.0.1:8000/v2/models/resnet18/infer"
var concurrencyLevels = []int{125, 130, 145}

// Request/Response structures for Triton
type TritonInput struct {
	Name     string          `json:"name"`
	Shape    []int           `json:"shape"`
	Datatype string          `json:"datatype"`
	Data     [][][][]float32 `json:"data"`
}

type TritonRequest struct {
	Inputs []TritonInput `json:"inputs"`
}

type TritonOutput struct {
	Shape []int `json:"shape"`
}

type TritonResponse struct {
	Outputs []TritonOutput `json:"outputs"`
}

type Result struct {
	ConcurrentRequests int
	RequestID          int
	InferenceTimeSec   float64
	OutputShape        string
}

func generateDummyData() TritonRequest {
	// Generate [1,3,224,224] random float32 data
	data := make([][][][]float32, 1)
	data[0] = make([][][]float32, 3)
	for c := 0; c < 3; c++ {
		data[0][c] = make([][]float32, 224)
		for h := 0; h < 224; h++ {
			data[0][c][h] = make([]float32, 224)
			for w := 0; w < 224; w++ {
				data[0][c][h][w] = rand.Float32()
			}
		}
	}
	return TritonRequest{
		Inputs: []TritonInput{
			{
				Name:     "input",
				Shape:    []int{1, 3, 224, 224},
				Datatype: "FP32",
				Data:     data,
			},
		},
	}
}

func hitTriton(client *http.Client, req TritonRequest, requestID int, concurrency int, results chan<- Result, wg *sync.WaitGroup) {
	defer wg.Done()

	// Serialize JSON
	body, _ := json.Marshal(req)

	start := time.Now()
	resp, err := client.Post(url, "application/json", bytes.NewBuffer(body))
	end := time.Now()

	duration := end.Sub(start).Seconds()
	if err != nil {
		fmt.Println("Error:", err)
		results <- Result{ConcurrentRequests: concurrency, RequestID: requestID, InferenceTimeSec: duration, OutputShape: "error"}
		return
	}
	defer resp.Body.Close()

	var tritonResp TritonResponse
	_ = json.NewDecoder(resp.Body).Decode(&tritonResp)

	shapeStr := "nil"
	if len(tritonResp.Outputs) > 0 {
		shapeStr = fmt.Sprintf("%v", tritonResp.Outputs[0].Shape)
	}

	results <- Result{
		ConcurrentRequests: concurrency,
		RequestID:          requestID,
		InferenceTimeSec:   duration,
		OutputShape:        shapeStr,
	}
}

func main() {
	allResults := []Result{}
	client := &http.Client{}

	for _, level := range concurrencyLevels {
		fmt.Printf("\nHitting %d concurrent requests...\n", level)

		req := generateDummyData()
		resultsChan := make(chan Result, level)
		var wg sync.WaitGroup
		wg.Add(level)

		for i := 0; i < level; i++ {
			go hitTriton(client, req, i+1, level, resultsChan, &wg)
		}

		wg.Wait()
		close(resultsChan)

		for r := range resultsChan {
			allResults = append(allResults, r)
			fmt.Printf("Request %d: time=%.4fs, output_shape=%s\n",
				r.RequestID, r.InferenceTimeSec, r.OutputShape)
		}
	}

	// Saving to the CSV
	file, err := os.Create("triton_async_results.csv")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	writer.Write([]string{"concurrent_requests", "request_id", "inference_time_sec", "output_shape"})
	for _, r := range allResults {
		writer.Write([]string{
			strconv.Itoa(r.ConcurrentRequests),
			strconv.Itoa(r.RequestID),
			fmt.Sprintf("%.6f", r.InferenceTimeSec),
			r.OutputShape,
		})
	}

	fmt.Println("\nResults saved to triton_async_results.csv")
}
