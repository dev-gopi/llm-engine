from inference.memory import LongTermMemory

def test_user_scoped_memory_and_delete(tmp_path):
    memory=LongTermMemory(tmp_path/'memory.sqlite')
    memory.add('u1','likes python programming',kind='semantic')
    memory.add('u2','likes cooking',kind='semantic')
    assert memory.retrieve('u1','python')[0].user_id=='u1'
    assert memory.retrieve('u1','cooking')==[]
    assert memory.delete_user('u1')==1
