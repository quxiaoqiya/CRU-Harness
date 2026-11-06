import unittest
from long_context_agent.visual_count_memory_v21 import normalize_counts, integer, run_count_memory


class CountNormalizationTests(unittest.TestCase):
    def test_full_anonymous_object_rows_are_bound_by_input_order(self):
        result=normalize_counts('[{"figures":1,"tables":0},{"figures":2,"tables":1}]',['b','a'])
        self.assertEqual(result,[{'id':'b','figures':1,'tables':0},{'id':'a','figures':2,'tables':1}])

    def test_short_anonymous_array_is_never_shifted_or_guessed(self):
        self.assertEqual(normalize_counts('[{"counts":[2,1]}]',['a','b']),[])

    def test_anonymous_nested_vectors_and_columns(self):
        expected=[{'id':'a','figures':2,'tables':0},{'id':'b','figures':1,'tables':3}]
        self.assertEqual(normalize_counts('[{"counts":[2,0]},{"counts":[1,3]}]',['a','b']),expected)
        self.assertEqual(normalize_counts('{"figures":[2,1],"tables":[0,3]}',['a','b']),expected)

    def test_explicit_ids_allow_partial_output_without_loss_of_alignment(self):
        self.assertEqual(normalize_counts('{"pages":[{"id":"b","figures":2,"tables":1}]}',['a','b']),
                         [{'id':'b','figures':2,'tables':1}])
        self.assertEqual(normalize_counts('[["b",2,1]]',['a','b']),[{'id':'b','figures':2,'tables':1}])

    def test_source_keyed_mapping_and_aliases(self):
        self.assertEqual(normalize_counts('{"b":{"picture_count":2,"table_count":1}}',['a','b']),
                         [{'id':'b','figures':2,'tables':1}])

    def test_unknown_sources_and_conflicting_aliases_are_rejected(self):
        self.assertEqual(normalize_counts('{"pages":[{"id":"x","figures":3,"tables":0}]}',['a']),[])
        self.assertEqual(normalize_counts('[{"figures":1,"picture_count":2,"tables":0}]',['a']),[])

    def test_numeric_normalization_does_not_turn_unknown_into_zero(self):
        for value in [True,None,-1,float('nan'),float('inf'),'unknown','1.5']:self.assertIsNone(integer(value))
        self.assertEqual(integer('2'),2);self.assertEqual(integer(2.0),2)

    def test_distinct_entity_counting_is_not_summed(self):
        with self.assertRaisesRegex(ValueError,'Non-additive'):
            run_count_memory('Count distinct people',None,[],None)

    def fake(self,outputs):
        class Client:
            def __init__(self):self.ledger=[];self.remaining_calls=100;self.messages=[]
            def can_call(self,*args,**kwargs):return self.remaining_calls>0
            def complete(self,messages,*args):
                self.messages.append(messages);self.ledger.append({});self.remaining_calls-=1
                value=outputs[len(self.ledger)-1]
                if isinstance(value,Exception):raise value
                return value
        class Index:
            pages={'a':{'id':'a'},'b':{'id':'b'}}
            def image(self,page):return 'data:image/png;base64,original-'+page['id']
        return Client(),Index()

    def test_partial_tagged_output_only_retries_missing_source(self):
        client,index=self.fake(['{"pages":[{"id":"b","figures":1,"tables":1}]}',
                                '{"pages":[{"id":"a","figures":2,"tables":0}]}'])
        result=run_count_memory('Count every picture and table',index,['a','b'],client)
        self.assertEqual(result['answer'],'4')
        self.assertEqual(result['trace'][1]['page_ids'],['a'])

    def test_nonprogressing_repair_terminates_after_source_attempt_bound(self):
        client,index=self.fake(['bad']*3)
        result=run_count_memory('Count every picture and table',index,['a','b'],client)
        self.assertEqual(result['answer'],'UNKNOWN');self.assertEqual(len(client.ledger),3)
        self.assertEqual(result['attempts'],{'a':3,'b':3})

    def test_uncertain_delivery_does_not_trigger_repair(self):
        client,index=self.fake([TimeoutError('delivery uncertain')])
        with self.assertRaises(TimeoutError):run_count_memory('Count every picture and table',index,['a','b'],client)
        self.assertEqual(len(client.ledger),1)

    def test_whole_scope_scalar_recovery_does_not_promote_a_page_answer(self):
        from long_context_agent.visual_count_outputs import explicit_global_answer
        self.assertEqual(explicit_global_answer('{"answer":42}'),42)
        self.assertEqual(explicit_global_answer('{"totals":{"figures":40,"tables":2}}'),42)
        self.assertIsNone(explicit_global_answer('{"pages":[{"id":"a","answer":2}]}'))
        self.assertIsNone(explicit_global_answer('{"id":"a","answer":2}'))
        self.assertIsNone(explicit_global_answer('[[2,1],[1,0]]'))

    def test_finished_pixels_are_evicted_but_count_facts_survive(self):
        client,index=self.fake(['{"pages":[{"id":"a","figures":1,"tables":0},{"id":"b","figures":2,"tables":0}]}'])
        index._images={'a':'pixels-a','b':'pixels-b'}
        result=run_count_memory('Count every picture and table',index,['a','b'],client)
        self.assertEqual(index._images,{})
        self.assertEqual(result['answer'],'3')
        self.assertEqual(len(result['aggregation']['facts']),2)


if __name__=='__main__':unittest.main()
