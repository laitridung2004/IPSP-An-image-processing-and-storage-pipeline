from pyspark.sql.streaming import StreamingQueryListener
import time
from datetime import datetime

class PerformanceListener(StreamingQueryListener):
    def __init__(self, query):
        self.query = query
        self.batch_stats = []
        self.start_time = time.time()
        self.last_batch_time = None
        self.total_rows_processed = 0
        self.has_received_data = False
        
        self.first_data_batch = None
        self.last_data_batch = None
        
        print("="*80)
        print(" Performance Listener Initialized")
        print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
    def onQueryStarted(self, event):
        print("\n" + "="*80)
        print(" SPARK STREAMING QUERY STARTED")
        print("="*80)
        print(f"   Query ID:   {event.id}")
        print(f"   Run ID:     {event.runId}")
        print(f"   Name:       {event.name if hasattr(event, 'name') else 'N/A'}")
        print(f"   Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")
        
    def onQueryProgress(self, event):
        progress = event.progress
        
        batch_id = progress.batchId
        num_input_rows = progress.numInputRows
        input_rows_per_sec = progress.inputRowsPerSecond
        process_rows_per_sec = progress.processedRowsPerSecond
        
        duration_ms = progress.durationMs
        trigger_execution = duration_ms.get('triggerExecution', 0)
        add_batch = duration_ms.get('addBatch', 0)
        get_batch = duration_ms.get('getBatch', 0)
        
        current_time = time.time()
        
        if num_input_rows == 0 and self.has_received_data:
            print("\n" + "="*40)
            print(" STOP CONDITION TRIGGERED")
            print("="*40)
            print(f"   Batch #{batch_id} received 0 rows")
            print(f"   This is the first empty batch after data stream")
            print(f"   Initiating graceful shutdown...")
            print("="*40 + "\n")
            
            self._print_final_summary(stopped_by_empty_batch=True)
            
            self.query.stop()
            return
        
        if num_input_rows > 0:
            if not self.has_received_data:
                self.has_received_data = True
                self.first_data_batch = batch_id
                print("\n" + "="*40)
                print(f" FIRST DATA RECEIVED at Batch #{batch_id}")
                print("="*40 + "\n")
            
            self.last_data_batch = batch_id
            self.total_rows_processed += num_input_rows
        
        print("-"*80)
        print(f" BATCH #{batch_id} @ {datetime.now().strftime('%H:%M:%S')}")
        print("-"*80)
        
        if num_input_rows > 0:
            print(f" Input:")
            print(f"   Rows received:           {num_input_rows:,}")
            if input_rows_per_sec:
                print(f"   Input rate:              {input_rows_per_sec:.2f} rows/sec")
            
            print(f"\n Processing:")
            if process_rows_per_sec:
                print(f"   Processing rate:         {process_rows_per_sec:.2f} rows/sec")
            print(f"   Trigger execution:       {trigger_execution:,}ms")
            print(f"   Add batch time:          {add_batch:,}ms")
            print(f"   Get batch time:          {get_batch:,}ms")
            
            print(f"\n  Detailed Timing:")
            for key in sorted(duration_ms.keys()):
                value = duration_ms[key]
                print(f"   {key:.<30} {value:>8,}ms")
            
            sources = progress.sources
            if sources:
                print(f"\n Kafka Source:")
                for source in sources:
                    print(f"   Input rows:              {source.numInputRows:,}")
                    if hasattr(source, 'startOffset') and source.startOffset:
                        try:
                            import json
                            offsets = json.loads(source.startOffset)
                            if 'traffic_data' in offsets:
                                partitions = offsets['traffic_data']
                                print(f"   Partitions processed:    {len(partitions)}")
                        except:
                            pass
            
            if process_rows_per_sec and input_rows_per_sec:
                if process_rows_per_sec < input_rows_per_sec * 0.8:
                    print(f"\n  WARNING: Processing slower than input!")
                    print(f"   Falling behind by:       {input_rows_per_sec - process_rows_per_sec:.2f} rows/sec")
        else:
            print(f" Empty batch (0 rows)")
            if not self.has_received_data:
                print(f"   Waiting for data from Producer...")
        
        if self.has_received_data:
            elapsed = current_time - self.start_time
            overall_rate = self.total_rows_processed / elapsed if elapsed > 0 else 0
            
            print(f"\n Cumulative Stats:")
            print(f"   Total batches:           {batch_id + 1}")
            print(f"   Data batches:            {batch_id - self.first_data_batch + 1}")
            print(f"   Total rows:              {self.total_rows_processed:,}")
            print(f"   Overall rate:            {overall_rate:.2f} rows/sec")
            print(f"   Runtime:                 {elapsed:.2f}s")
        
        print("-"*80 + "\n")
        
        batch_stat = {
            'batch_id': batch_id,
            'timestamp': current_time,
            'num_input_rows': num_input_rows,
            'input_rate': input_rows_per_sec,
            'process_rate': process_rows_per_sec,
            'trigger_execution_ms': trigger_execution,
            'add_batch_ms': add_batch,
            'get_batch_ms': get_batch,
            'total_duration_ms': sum(duration_ms.values())
        }
        self.batch_stats.append(batch_stat)
        
        self.last_batch_time = current_time
    
    def onQueryTerminated(self, event):
        print("\n" + "="*80)
        print(" SPARK STREAMING QUERY TERMINATED")
        print("="*80)
        print(f"   Query ID:   {event.id}")
        print(f"   Run ID:     {event.runId}")
        
        if hasattr(event, 'exception') and event.exception:
            print(f"   Exception:  {event.exception}")
            print("="*80 + "\n")
        else:
            print(f"   Status:     Clean shutdown")
            print("="*80 + "\n")
            
            if not hasattr(self, '_final_summary_printed'):
                self._print_final_summary(stopped_by_empty_batch=False)
    
    def _print_final_summary(self, stopped_by_empty_batch=False):
        self._final_summary_printed = True
        
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print(" FINAL PERFORMANCE SUMMARY")
        print("="*80)
        
        print(f"\n Overall Statistics:")
        print(f"   Total runtime:           {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"   Total batches:           {len(self.batch_stats)}")
        
        if self.has_received_data:
            data_batches = [b for b in self.batch_stats if b['num_input_rows'] > 0]
            empty_batches = [b for b in self.batch_stats if b['num_input_rows'] == 0]
            
            print(f"   Data batches:            {len(data_batches)}")
            print(f"   Empty batches:           {len(empty_batches)}")
            print(f"   First data batch:        #{self.first_data_batch}")
            print(f"   Last data batch:         #{self.last_data_batch}")
            
            print(f"\n Data Processing:")
            print(f"   Total rows processed:    {self.total_rows_processed:,}")
            
            if total_time > 0:
                overall_throughput = self.total_rows_processed / total_time
                print(f"   Overall throughput:      {overall_throughput:.2f} rows/sec")
            
            if data_batches:
                avg_batch_size = self.total_rows_processed / len(data_batches)
                print(f"   Average batch size:      {avg_batch_size:.2f} rows")
            
            print(f"\n  Timing Statistics:")
            
            total_trigger_time = sum(b['trigger_execution_ms'] for b in data_batches)
            total_add_batch = sum(b['add_batch_ms'] for b in data_batches)
            total_get_batch = sum(b['get_batch_ms'] for b in data_batches)
            
            print(f"   Total trigger execution: {total_trigger_time:,}ms ({total_trigger_time/1000:.2f}s)")
            print(f"   Total add batch time:    {total_add_batch:,}ms ({total_add_batch/1000:.2f}s)")
            print(f"   Total get batch time:    {total_get_batch:,}ms ({total_get_batch/1000:.2f}s)")
            
            if data_batches:
                avg_trigger = total_trigger_time / len(data_batches)
                avg_add_batch = total_add_batch / len(data_batches)
                avg_get_batch = total_get_batch / len(data_batches)
                
                print(f"\n   Average per batch:")
                print(f"      Trigger execution:    {avg_trigger:.2f}ms")
                print(f"      Add batch:            {avg_add_batch:.2f}ms")
                print(f"      Get batch:            {avg_get_batch:.2f}ms")
            
            print(f"\n Processing Rates:")
            
            rates_with_data = [b['process_rate'] for b in data_batches if b['process_rate']]
            if rates_with_data:
                avg_process_rate = sum(rates_with_data) / len(rates_with_data)
                max_process_rate = max(rates_with_data)
                min_process_rate = min(rates_with_data)
                
                print(f"   Average processing rate: {avg_process_rate:.2f} rows/sec")
                print(f"   Peak processing rate:    {max_process_rate:.2f} rows/sec")
                print(f"   Min processing rate:     {min_process_rate:.2f} rows/sec")
            
            print(f"\n Batch Details:")
            print(f"   {'Batch':<8} {'Rows':<10} {'Process Rate':<15} {'Duration':<12}")
            print(f"   {'-'*8} {'-'*10} {'-'*15} {'-'*12}")
            
            for batch in data_batches[:10]:
                batch_id = batch['batch_id']
                rows = batch['num_input_rows']
                rate = batch['process_rate'] if batch['process_rate'] else 0
                duration = batch['trigger_execution_ms']
                
                print(f"   #{batch_id:<7} {rows:<10,} {rate:<15.2f} {duration:<12,}ms")
            
            if len(data_batches) > 10:
                print(f"   ... and {len(data_batches) - 10} more batches")
            
            print(f"\n Termination:")
            if stopped_by_empty_batch:
                print(f"   Reason:                  First empty batch detected")
                print(f"   Auto-stop:                Enabled")
            else:
                print(f"   Reason:                  Manual/External stop")
        else:
            print(f"\n  No data was received during this run")
            print(f"   Producer may not have started")
            print(f"   Or Kafka topic is empty")
        
        print("="*80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80 + "\n")